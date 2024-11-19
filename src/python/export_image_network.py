import argparse
import subprocess
import time
from pathlib import Path

import onnx
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet3d.models.necks.view_transformer import ASPP
from mmdet.models.backbones.resnet import BasicBlock
from onnxsim import simplify
from tools.misc.fuse_conv_bn import fuse_module
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--samples", default=400, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--no-acceleration",
        action="store_true",
        help="Omit the pre-computation acceleration",
    )
    args = parser.parse_args()
    return args


def export_simplify_optimize_onnx(
    onnx_output_prefix,
    model,
    inputs,
    input_names=None,
    output_names=None,
    opset_version=14,
):
    # make dir
    onnx_output_prefix = Path(onnx_output_prefix)
    onnx_output_prefix.mkdir(parents=True, exist_ok=True)
    onnx_output_prefix = str(onnx_output_prefix / onnx_output_prefix.name)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        args=inputs,
        f=f"{onnx_output_prefix}.onnx",
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    # Simplify the model
    model_simp, check = simplify(f"{onnx_output_prefix}.onnx")
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, f"{onnx_output_prefix}_simp.onnx")
    print(f">> Exported ONNX model to {onnx_output_prefix}_simp.onnx")

    # Optimize the simplified model
    process = subprocess.run(
        [
            "python3",
            "-m",
            "onnxoptimizer",
            f"{onnx_output_prefix}_simp.onnx",
            f"{onnx_output_prefix}_opt.onnx",
        ],
        check=False,
    )

    if process.returncode == 0:
        print(f">> Exported ONNX model to {onnx_output_prefix}_opt.onnx")
        return True
    else:
        print("onnxoptimizer execution has failed with exit code:", process.returncode)
        return False


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.data.test["data_root"] = "/data/nuscenes"
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    data = next(iter(data_loader))
    for key in data.keys():
        print(f"{key=}, {type(data[key][0])=} {len(data[key])}")

    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate = True
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    bev_encoder_input = {}
    def save_outputs_hook(module, input):
        bev_encoder_input["bev_encoder"] = input[0].detach()
    model.module.img_bev_encoder_backbone.register_forward_pre_hook(save_outputs_hook)

    inputs = [d.cuda() for d in data["img_inputs"][0]]
    with torch.no_grad():
        feat_prev, inputs = model.module.extract_img_feat(
            inputs, pred_prev=True, img_metas=None
        )
    data["img_inputs"][0] = inputs

    print("prepared data for inference...")

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        model(
            return_loss=False,
            rescale=True,
            sequential=True,
            feat_prev=feat_prev,
            **data,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print(f"Elapsed time: {elapsed}s fps: {1/elapsed}")

    class ImgNn(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.D = self.model.module.img_view_transformer.D
            self.out_channels = self.model.module.img_view_transformer.out_channels

        def forward(self, imgs, mlp_input):
            N, C, imH, imW = imgs.shape
            imgs = imgs.view(N, C, imH, imW)
            x = self.model.module.img_backbone(imgs)
            if self.model.module.with_img_neck:
                x = self.model.module.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
            _, output_dim, output_H, output_W = x.shape
            x = x.view(N, output_dim, output_H, output_W)

            N, C, H, W = x.shape
            x = x.view(N, C, H, W)
            x = self.model.module.img_view_transformer.depth_net(x, mlp_input)

            depth_digit = x[:, : self.D, ...]
            tran_feat = x[:, self.D : self.D + self.out_channels, ...]
            depth = depth_digit.softmax(dim=1)

            feat = tran_feat.view(N, self.out_channels, H, W)
            feat = feat.permute(0, 2, 3, 1)
            depth = depth.view(N, self.D, H, W)
            return depth, feat

        def prepare_mlp_input(self, inputs):
            _, rots_curr, trans_curr, intrins = inputs[:4]
            _, _, post_rots, post_trans, bda = inputs[4:]

            mlp_input = self.model.module.img_view_transformer.get_mlp_input(
                rots_curr[0:1, ...],
                trans_curr[0:1, ...],
                intrins,
                post_rots,
                post_trans,
                bda[0:1, ...],
            )
            return mlp_input

        def extract_img_feat_sequential(self, inputs, mlp_input):
            imgs, rots_curr, trans_curr, intrins = inputs[:4]
            _, _, post_rots, post_trans, bda = inputs[4:]

            return self.forward(imgs, mlp_input)

    class ImgBbNeckResnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.D = self.model.module.img_view_transformer.D
            self.out_channels = self.model.module.img_view_transformer.out_channels
            self.depth_net = self.model.module.img_view_transformer.depth_net
            self.conv = nn.Conv2d(
                512, 80, kernel_size=1, stride=1, padding=0, bias=True
            )

            mid_channels = 512
            depth_channels = 118
            depth_conv_list = [
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
            ]
            self.depth_conv = nn.Sequential(*depth_conv_list)

        def forward(self, imgs):
            # image network backbone
            N, C, imH, imW = imgs.shape
            imgs = imgs.view(N, C, imH, imW)
            x = self.model.module.img_backbone(imgs)
            if self.model.module.with_img_neck:
                x = self.model.module.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
            _, output_dim, output_H, output_W = x.shape
            x = x.view(N, output_dim, output_H, output_W)
            N, C, H, W = x.shape
            x = x.view(N, C, H, W)

            # from here, the depthnet convolutions are applied
            x = self.depth_net.reduce_conv(x)
            self.conv.to(x.device)
            feat = self.conv(x)
            self.depth_conv.to(x.device)
            depth = self.depth_conv(x)
            return feat, depth

    class ImgNnOnline(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.depth_net = self.model.module.img_view_transformer.depth_net
            self.D = self.model.module.img_view_transformer.D
            self.out_channels = self.model.module.img_view_transformer.out_channels

        def offline_input(self, mlp_input):
            mlp_input = self.depth_net.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.depth_net.context_mlp(mlp_input)[..., None, None]
            depth_se = self.depth_net.depth_mlp(mlp_input)[..., None, None]
            return context_se, depth_se

        def forward(self, imgs, context_se, depth_se):
            N, C, imH, imW = imgs.shape
            imgs = imgs.view(N, C, imH, imW)
            x = self.model.module.img_backbone(imgs)
            if self.model.module.with_img_neck:
                x = self.model.module.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
            _, output_dim, output_H, output_W = x.shape
            x = x.view(N, output_dim, output_H, output_W)

            N, C, H, W = x.shape
            x = x.view(N, C, H, W)
            x = self.depth_net.reduce_conv(x)
            context = self.depth_net.context_se(x, context_se)
            context = self.depth_net.context_conv(context)
            depth = self.depth_net.depth_se(x, depth_se)
            depth = self.depth_net.depth_conv(depth)
            return torch.cat([depth, context], dim=1)

    class DepthnetWoAspp(nn.Module):
        def __init__(self):
            super().__init__()
            mid_channels = 512
            depth_channels = 118
            depth_conv_list = [
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
            ]
            # depth_conv_list.append(ASPP(mid_channels, mid_channels))
            depth_conv_list.append(
                nn.Conv2d(
                    mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
                )
            )
            self.depth_conv = nn.Sequential(*depth_conv_list)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(x)  # necessary for front-end
            x = self.depth_conv(x)
            return x

    class AsppTest(nn.Module):
        def __init__(self):
            super().__init__()
            mid_channels = 512
            depth_channels = 118
            self.img_aspp = ASPP(mid_channels, mid_channels)
            self.conv = nn.Conv2d(
                mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
            )

        def forward(self, x):
            x = self.img_aspp(x)
            x = self.conv(x)
            return x
    
    class BEVEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv = nn.Conv2d(720, 720, kernel_size=1, stride=1, padding=0, bias=True)
            self.img_bev_encoder_backbone = model.module.img_bev_encoder_backbone
            self.img_bev_encoder_neck = model.module.img_bev_encoder_neck

        def forward(self, x):
            self.conv.to(x.device)
            x = self.conv(x)
            x = self.img_bev_encoder_backbone(x)
            x = self.img_bev_encoder_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
            return x


    img_nn = ImgNn(model)
    img_nn.eval()

    img_backbone = ImgBbNeckResnet(model)
    img_backbone.eval()

    img_depthnet_wo_aspp = DepthnetWoAspp()
    img_depthnet_wo_aspp.eval()

    img_aspp = AsppTest()
    img_aspp.eval()

    bev_encoder = BEVEncoder(model)
    bev_encoder.eval()
    
    with torch.no_grad():
        mlp_input = img_nn.prepare_mlp_input(inputs)
        mlp_input = mlp_input[0]
        imgs = inputs[0][0]
        print(f"{imgs.shape=}, {mlp_input.shape=}")

        depth, feat = img_nn(imgs, mlp_input)
        
        # if depth.npy and feat.npy are present, compare the results
        if Path("depth.npy").exists() and Path("feat.npy").exists():
            import numpy as np

            ref_depth = np.load("depth.npy")
            ref_feat = np.load("feat.npy")

            print(f"{np.allclose(depth.cpu().numpy(), ref_depth)=}")
            print(f"{np.allclose(feat.cpu().numpy(), ref_feat)=}")
        onnx_output_prefix = "img_nn"
        export_simplify_optimize_onnx(
            onnx_output_prefix,
            img_nn,
            (imgs, mlp_input),
            input_names=["imgs", "mlp_input"],
            output_names=["depth", "feat"],
        )

        onnx_output_prefix = "img_backbone"
        export_simplify_optimize_onnx(
            onnx_output_prefix,
            img_backbone,
            (imgs,),
            input_names=["imgs"],
            output_names=["feat", "depth"],
        )

        onnx_output_prefix = "img_depthnet_wo_aspp"
        x = torch.randn(1, 512, 16, 44)
        export_simplify_optimize_onnx(
            onnx_output_prefix,
            img_depthnet_wo_aspp,
            (x,),
            input_names=["x"],
            output_names=["out"],
        )

        onnx_output_prefix = "img_aspp"
        x = torch.randn(1, 512, 16, 44)
        export_simplify_optimize_onnx(
            onnx_output_prefix,
            img_aspp,
            (x,),
            input_names=["x"],
            output_names=["out"],
        )
        import numpy as np 
        np.save("bev_encoder_input.npy", bev_encoder_input["bev_encoder"].cpu().numpy())
        onnx_output_prefix = "bev_encoder"
        export_simplify_optimize_onnx(
            onnx_output_prefix,
            bev_encoder,
            (bev_encoder_input["bev_encoder"],),
            input_names=["input"],
            output_names=["output"],
        )


if __name__ == "__main__":
    main()
