import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module
from vis_utils import vis
from mmdet3d.core import bbox3d2result
import numpy as np 
import pickle
from torchmetrics.image import PeakSignalNoiseRatio

class HeadPostProcessor(torch.nn.Module):
    def __init__(self, model):
        super(HeadPostProcessor, self).__init__()
        self.pts_bbox_head = model.module.pts_bbox_head

    def forward(self, outs, img_metas = None, rescale = True):
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list


class BEVDETDrawer:

    def __init__(self):
        configs = "/workspace/bevdet/bevdet/configs/bevdet/bevdet4d-r50-depth-cbgs.py"
        check_point = "/workspace/bevdet/src/python/bevdet4d-r50-depth-cbgs.pth"
        cfg = Config.fromfile(configs)
        parser = argparse.ArgumentParser()
        parser = vis.parse_args(parser)
        self.args = parser.parse_args()
        cfg.data.test.test_mode = True
    # * build model
        cfg.model.train_cfg = None
        cfg.model.align_after_view_transfromation = True
        cfg.model.img_view_transformer.accelerate = True
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        model = MMDataParallel(model, device_ids=[0])
        self.head_post_processor = HeadPostProcessor(model)

        # cfg.data.test["data_root"] = "data/nuscenes"
        self.dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )
        data = next(iter(data_loader))
        #dhe

        self.output_dump = np.load("/workspace/bevdet/src/python/output_dump.npy", allow_pickle=True)
        # self.args.root_path = "data/nuscenes"
        
        info_path = self.args.root_path + "/bevdetv2-nuscenes_infos_%s.pkl" % self.args.version
        # info_path = "/workspace/bevdet/bevdet/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl" % self.args.version
        self.dataset_vis = pickle.load(open(info_path, "rb"))
        self.img_meta = data["img_metas"][0].data[0]
        self.sample_token = self.img_meta[0]["sample_idx"]

        # # infos 리스트를 딕셔너리로 변환 (초기화 시 한 번 수행)
        # if not hasattr(self, 'infos_dict'):
        #     self.infos_dict = {info["token"]: info for info in self.dataset_vis["infos"]}

        self.infos = None
        for info in self.dataset_vis["infos"]:
            if info["token"] == self.sample_token:
                self.infos = info
                break
        self.args.scale_factor = 1

            
    def get_det_vis(self, det_dump, input_imgs, show_psnr = False):
        # det_dump = np.load("/bevdet/det_result.npz", allow_pickle=True)
        det_results = [{} for i in range(6)]
        for key in det_dump:
            data_type, idx = key.split("_")
            output_ = torch.frombuffer(det_dump[key], dtype=torch.float32).cuda()
            N, C, H, W = self.output_dump[int(idx)][0][data_type].shape
            output_ = output_.reshape((N, H, W, C))
            output_ = output_.permute(0, 3, 1, 2)
            det_results[int(idx)][data_type] = output_

        det_results = [np.array([d]) for d in det_results]

        if show_psnr:
            for idx in range(6):
                for key in self.output_dump[idx][0].keys():
                    output_ref = self.output_dump[idx][0][key].to('cpu')
                    output_preds = det_results[idx][0][key].to('cpu')
                    psnr = PeakSignalNoiseRatio()
                    psnr_output = psnr(output_preds, output_ref)
                    print(f"{idx}-{key}: {psnr_output:.3f}")

        results = self.head_post_processor(det_results, self.img_meta)
        nusc_submissions = vis.format_single_result(self.sample_token, self.dataset, results[0])   
        input_imgs = np.pad(input_imgs, ((0, 0), (180, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        output_img = vis.visualize_acc(self.args, self.sample_token, nusc_submissions, self.infos, input_imgs)
        return output_img
        # return None

    def get_det_vis_moon1(self, det_dump, img_idx, show_psnr = False):
        # det_dump = np.load("/bevdet/det_result.npz", allow_pickle=True)
        det_results = [{} for i in range(6)]
        # img_inputs1 = self.img_inputs.squeeze(0)[0:6,:,:,:].permute(0,2,3,1)

        # self.img_display(input_imgs,img_inputs1)
        
        # self.img_display(input_imgs,img_inputs1)

        # ########### dhe ########### 
        # # print ('frame_counter:', img_idx)
        # self.img_meta = self.dataset[img_idx]["img_metas"][0].data
        # self.sample_token = self.img_meta["sample_idx"]

        # # # 딕셔너리를 통해 빠르게 접근
        # # self.infos = self.infos_dict.get(self.sample_token)
        
        # self.infos = None
        # for info in self.dataset_vis["infos"]:
        #     if info["token"] == self.sample_token:
        #         self.infos = info
        #         break
        
        ####
        for key in det_dump:
            data_type, idx = key.split("_")

            # 쓰기 가능한 버퍼로 복사
            # buffer_copy = np.copy(det_dump[key])
            # output_ = torch.frombuffer(buffer_copy, dtype=torch.float32)
            output_ = torch.frombuffer(det_dump[key], dtype=torch.float32) # legacy code
            
            N, C, H, W = self.output_dump[int(idx)][0][data_type].shape
            output_ = output_.reshape((N, H, W, C))
            output_ = output_.permute(0, 3, 1, 2)
            
            # clone().detach()로 새로운 텐서 생성
            # det_results[int(idx)][data_type] = output_.clone().detach()
            det_results[int(idx)][data_type] = output_  # legacy code
 
        det_results = [np.array([d]) for d in det_results]

        if show_psnr:
            for idx in range(6):
                for key in self.output_dump[idx][0].keys():
                    output_ref = self.output_dump[idx][0][key].to('cpu')
                    output_preds = det_results[idx][0][key].to('cpu')
                    psnr = PeakSignalNoiseRatio()
                    psnr_output = psnr(output_preds, output_ref)
                    print(f"{idx}-{key}: {psnr_output:.3f}")

        results = self.head_post_processor(det_results, self.img_meta) # img_meta 리스트 조심 ~~
        nusc_submissions = vis.format_single_result(self.sample_token, self.dataset, results[0])   
        # input_imgs = np.pad(input_imgs, ((0, 0), (180, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        # output_img = vis.visualize_acc(self.args, self.sample_token, nusc_submissions, self.infos, input_imgs)
        output_img = vis.visualize_acc_moon1(self.args, self.sample_token, nusc_submissions , self.infos, img_idx)
        return output_img
    
    def get_det_vis_moon(self, det_dump, show_psnr = False):
        # det_dump = np.load("/bevdet/det_result.npz", allow_pickle=True)
        det_results = [{} for i in range(6)]
        for key in det_dump:
            data_type, idx = key.split("_")
            
            # 쓰기 가능한 버퍼로 복사
            # buffer_copy = np.copy(det_dump[key])
            # output_ = torch.frombuffer(buffer_copy, dtype=torch.float32)
            output_ = torch.frombuffer(det_dump[key], dtype=torch.float32) # legacy code
            
            N, C, H, W = self.output_dump[int(idx)][0][data_type].shape
            output_ = output_.reshape((N, H, W, C))
            output_ = output_.permute(0, 3, 1, 2)
            
            # clone().detach()로 새로운 텐서 생성
            # det_results[int(idx)][data_type] = output_.clone().detach()
            det_results[int(idx)][data_type] = output_  # legacy code
 
        det_results = [np.array([d]) for d in det_results]

        if show_psnr:
            for idx in range(6):
                for key in self.output_dump[idx][0].keys():
                    output_ref = self.output_dump[idx][0][key].to('cpu')
                    output_preds = det_results[idx][0][key].to('cpu')
                    psnr = PeakSignalNoiseRatio()
                    psnr_output = psnr(output_preds, output_ref)
                    print(f"{idx}-{key}: {psnr_output:.3f}")

        results = self.head_post_processor(det_results, self.img_meta)
        nusc_submissions = vis.format_single_result(self.sample_token, self.dataset, results[0])   
        # input_imgs = np.pad(input_imgs, ((0, 0), (180, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        # output_img = vis.visualize_acc(self.args, self.sample_token, nusc_submissions, self.infos, input_imgs)
        output_img = vis.visualize_acc_moon(self.args, self.sample_token, nusc_submissions, self.infos)
        return output_img