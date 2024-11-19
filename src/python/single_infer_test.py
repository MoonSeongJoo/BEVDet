import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module
from utils import vis


def parse_args():
    parser = argparse.ArgumentParser(description="Sindle Inference Test")
    # * Inference
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

    parser = vis.parse_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # * set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # * build model
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate = True
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        print(f"load model from: {args.checkpoint}")
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # * set input data
    cfg.data.test["data_root"] = "./data/nuscenes"
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # check index = 721 for checking performance
    data = next(iter(data_loader))  # it is just index 0

    inputs = [d.cuda() for d in data["img_inputs"][0]]
    with torch.no_grad():
        feat_prev, inputs = model.module.extract_img_feat(
            inputs, pred_prev=True, img_metas=None
        )
    data["img_inputs"][0] = inputs
    print("prepared data for inference...")

    print("\n< Step.1 Inference >\n")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        results = model(
            return_loss=False,
            rescale=True,
            sequential=True,
            feat_prev=feat_prev,
            **data,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed}s fps: {1/elapsed}")

    if args.checkpoint is None:
        return

    print("\n< Step.2 Formatting >\n")
    sample_token = data["img_metas"][0].data[0][0]["sample_idx"]
    nusc_submissions = vis.format_single_result(sample_token, dataset, results[0])

    print("\n< Step.3 Visualization >\n")
    vis.visualize(args, sample_token, nusc_submissions)


if __name__ == "__main__":
    main()
