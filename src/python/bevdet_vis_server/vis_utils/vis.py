import pickle
from pathlib import Path

import cv2
import numpy as np
import pyquaternion
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion.quaternion import Quaternion
import torch
import torchvision.transforms.functional as TF


def parse_args(parser):
    # * Visualization
    parser.add_argument(
        "--show-range", type=int, default=50, help="Range of visualization in BEV"
    )
    parser.add_argument(
        "--canva-size", type=int, default=1000, help="Size of canva in pixel"
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Trade-off between image-view and bev in size of " "the visualized canvas",
    )
    parser.add_argument(
        "--vis-thred", type=float, default=0.3, help="Threshold the predicted results"
    )
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument(
        "--version", type=str, default="val", help="Version of nuScenes dataset"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        # default="/data/nuscenes",
        default="data/nuscenes",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/vis",
        help="Path to save visualization results",
    )

    return parser


color_map = {0: (255, 255, 0), 1: (0, 255, 255)}
draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
views = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = infos["lidar2ego_translation"]
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(infos["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = infos["ego2global_translation"]
    return ego2global @ lidar2ego


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)],
        axis=1,
    )
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info["sensor2lidar_rotation"]
    camera2lidar[:3, 3] = camrera_info["sensor2lidar_translation"]
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info["cam_intrinsic"]
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height)
    )
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [
            [max_lumi, 0, max_lumi],
            [max_lumi, 0, 0],
            [max_lumi, max_lumi, 0],
            [0, max_lumi, 0],
            [0, max_lumi, max_lumi],
            [0, 0, max_lumi],
        ],
        dtype=np.float32,
    )
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple((colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def format_single_result(sample_token, dataset, result):
    sample_id = None
    for i, info in enumerate(dataset.data_infos):
        if info["token"] == sample_token:
            sample_id = i
            break
    mapped_class_names = dataset.CLASSES
    boxes = result["pts_bbox"]["boxes_3d"].tensor.numpy()
    scores = result["pts_bbox"]["scores_3d"].numpy()
    labels = result["pts_bbox"]["labels_3d"].numpy()
    trans = dataset.data_infos[sample_id]["cams"][dataset.ego_cam][
        "ego2global_translation"
    ]
    rot = dataset.data_infos[sample_id]["cams"][dataset.ego_cam]["ego2global_rotation"]
    rot = pyquaternion.Quaternion(rot)
    nusc_annos = {}
    annos = list()
    for i, box in enumerate(boxes):
        name = mapped_class_names[labels[i]]
        center = box[:3]
        wlh = box[[4, 3, 5]]
        box_yaw = box[6]
        box_vel = box[7:].tolist()
        box_vel.append(0)
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
        nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
        nusc_box.rotate(rot)
        nusc_box.translate(trans)
        if np.sqrt(nusc_box.velocity[0] ** 2 + nusc_box.velocity[1] ** 2) > 0.2:
            if name in [
                "car",
                "construction_vehicle",
                "bus",
                "truck",
                "trailer",
            ]:
                attr = "vehicle.moving"
            elif name in ["bicycle", "motorcycle"]:
                attr = "cycle.with_rider"
            else:
                attr = dataset.DefaultAttribute[name]
        else:
            if name in ["pedestrian"]:
                attr = "pedestrian.standing"
            elif name in ["bus"]:
                attr = "vehicle.stopped"
            else:
                attr = dataset.DefaultAttribute[name]
        nusc_anno = dict(
            sample_token=sample_token,
            translation=nusc_box.center.tolist(),
            size=nusc_box.wlh.tolist(),
            rotation=nusc_box.orientation.elements.tolist(),
            velocity=nusc_box.velocity[:2],
            detection_name=name,
            detection_score=float(scores[i]),
            attribute_name=attr,
        )
        annos.append(nusc_anno)
    # other views results of the same frame should be concatenated
    if sample_token in nusc_annos:
        nusc_annos[sample_token].extend(annos)
    else:
        nusc_annos[sample_token] = annos
    nusc_submissions = {
        "meta": dataset.modality,
        "results": nusc_annos,
    }

    return nusc_submissions


def visualize(args, sample_token, nusc_submissions):
    info_path = args.root_path + "/bevdetv2-nuscenes_infos_%s.pkl" % args.version
    dataset = pickle.load(open(info_path, "rb"))
    vis_dir = Path(args.save_path)
    vis_dir.mkdir(parents=True, exist_ok=True)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    infos = None
    for info in dataset["infos"]:
        if info["token"] == sample_token:
            infos = info
            break

    pred_res = nusc_submissions["results"][infos["token"]]

    pred_boxes = [
        pred_res[rid]["translation"]
        + pred_res[rid]["size"]
        + [Quaternion(pred_res[rid]["rotation"]).yaw_pitch_roll[0] + np.pi / 2]
        for rid in range(len(pred_res))
    ]
    if len(pred_boxes) == 0:
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
    else:
        pred_boxes = np.array(pred_boxes, dtype=np.float32)
        boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
        corners_global = boxes.corners.numpy().reshape(-1, 3)
        corners_global = np.concatenate(
            [corners_global, np.ones([corners_global.shape[0], 1])], axis=1
        )
        l2g = get_lidar2global(infos)
        corners_lidar = corners_global @ np.linalg.inv(l2g).T
        corners_lidar = corners_lidar[:, :3]
    pred_flag = np.ones((corners_lidar.shape[0] // 8,), dtype=bool)
    scores = [pred_res[rid]["detection_score"] for rid in range(len(pred_res))]
    if args.draw_gt:
        gt_boxes = infos["gt_boxes"]
        gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
        width = gt_boxes[:, 4].copy()
        gt_boxes[:, 4] = gt_boxes[:, 3]
        gt_boxes[:, 3] = width
        corners_lidar_gt = (
            LB(infos["gt_boxes"], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        )
        corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)
        gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
        pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)
        scores = scores + [0 for _ in range(infos["gt_boxes"].shape[0])]
    scores = np.array(scores, dtype=np.float32)
    sort_ids = np.argsort(scores)

    # image view
    imgs = []
    for view in views:
        img = cv2.imread(infos["cams"][view]["data_path"])
        # draw instances
        corners_img, valid = lidar2img(corners_lidar, infos["cams"][view])
        valid = np.logical_and(
            valid, check_point_in_img(corners_img, img.shape[0], img.shape[1])
        )
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(int)
        for aid in range(valid.shape[0]):
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(
                        img,
                        corners_img[aid, index[0]],
                        corners_img[aid, index[1]],
                        color=color_map[int(pred_flag[aid])],
                        thickness=scale_factor,
                    )
        imgs.append(img)

    # bird-eye-view
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
    # draw lidar points
    lidar_points = np.fromfile(infos["lidar_path"], dtype=np.float32)
    lidar_points = lidar_points.reshape(-1, 5)[:, :3]
    lidar_points[:, 1] = -lidar_points[:, 1]
    lidar_points[:, :2] = (
        (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
    )
    for p in lidar_points:
        if check_point_in_img(p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
            color = depth2color(p[2])
            cv2.circle(
                canvas, (int(p[0]), int(p[1])), radius=0, color=color, thickness=1
            )

    # draw instances
    corners_lidar = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = (
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    )
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)

    for rid in sort_ids:
        score = scores[rid]
        if score < args.vis_thred and pred_flag[rid]:
            continue
        score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
        color = color_map[int(pred_flag[rid])]
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                [color[0] * score, color[1] * score, color[2] * score],
                thickness=1,
            )
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            [color[0] * score, color[1] * score, color[2] * score],
            1,
            lineType=8,
        )

    # fuse image-view and bev
    img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
    img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate(
        [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1
    )
    img[900 + canva_size * scale_factor :, :, :] = img_back
    img = cv2.resize(
        img,
        (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)),
    )
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    img[
        int(900 / scale_factor) : int(900 / scale_factor) + canva_size,
        w_begin : w_begin + canva_size,
        :,
    ] = canvas

    save_path = vis_dir / f"{infos['token']}.jpg"
    cv2.imwrite(str(save_path), img)
    print(f"saved visualized result to {save_path}")



def visualize_acc(args, sample_token, nusc_submissions, infos, input_imgs):

    vis_dir = Path(args.save_path)
    vis_dir.mkdir(parents=True, exist_ok=True)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    pred_res = nusc_submissions["results"][infos["token"]]

    pred_boxes = [
        pred_res[rid]["translation"]
        + pred_res[rid]["size"]
        + [Quaternion(pred_res[rid]["rotation"]).yaw_pitch_roll[0] + np.pi / 2]
        for rid in range(len(pred_res))
    ]
    
    if len(pred_boxes) == 0:
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
    else:
        pred_boxes = np.array(pred_boxes, dtype=np.float32)
        boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
        corners_global = boxes.corners.numpy().reshape(-1, 3)
        corners_global = np.concatenate(
            [corners_global, np.ones([corners_global.shape[0], 1])], axis=1
        )
        l2g = get_lidar2global(infos)
        corners_lidar = corners_global @ np.linalg.inv(l2g).T
        corners_lidar = corners_lidar[:, :3]
    pred_flag = np.ones((corners_lidar.shape[0] // 8,), dtype=bool)
    scores = [pred_res[rid]["detection_score"] for rid in range(len(pred_res))]
    if args.draw_gt:
        gt_boxes = infos["gt_boxes"]
        gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
        width = gt_boxes[:, 4].copy()
        gt_boxes[:, 4] = gt_boxes[:, 3]
        gt_boxes[:, 3] = width
        corners_lidar_gt = (
            LB(infos["gt_boxes"], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        )
        corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)
        gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
        pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)
        scores = scores + [0 for _ in range(infos["gt_boxes"].shape[0])]
    scores = np.array(scores, dtype=np.float32)
    sort_ids = np.argsort(scores)

    # image view
    imgs = []
    for idx, view in enumerate(views):
        # img = cv2.imread(infos["cams"][view]["data_path"])
        img = input_imgs[idx]
        img = cv2.resize(img, (1600, 900))
        # draw instances
        corners_img, valid = lidar2img(corners_lidar, infos["cams"][view])
        valid = np.logical_and(
            valid, check_point_in_img(corners_img, img.shape[0], img.shape[1])
        )
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(int)
        for aid in range(valid.shape[0]):
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(
                        img,
                        corners_img[aid, index[0]],
                        corners_img[aid, index[1]],
                        color=color_map[int(pred_flag[aid])],
                        thickness=scale_factor,
                    )
        imgs.append(img)

    # bird-eye-view
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
    
    # draw instances
    corners_lidar = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = (
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    )
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)

    for rid in sort_ids:
        score = scores[rid]
        if score < args.vis_thred and pred_flag[rid]:
            continue
        score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
        color = color_map[int(pred_flag[rid])]
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                [color[0] * score, color[1] * score, color[2] * score],
                thickness=1,
            )
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            [color[0] * score, color[1] * score, color[2] * score],
            1,
            lineType=8,
        )

    # fuse image-view and bev
    img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
    img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate(
        [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1
    )
    img[900 + canva_size * scale_factor :, :, :] = img_back
    img = cv2.resize(
        img,
        (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)),
    )
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    img[
        int(900 / scale_factor) : int(900 / scale_factor) + canva_size,
        w_begin : w_begin + canva_size,
        :,
    ] = canvas

    return img

def torch_jit_load(img_idx):
    # 파일 경로와 프레임 카운트를 설정합니다.
    frame_count = img_idx
    file_path = f"/workspace/bevdet/src/python/testset_03_pth/temp/input_data_{frame_count % 39}.pth"
    # 경로 /workspace/bevdet/src/python/testset_03_pth/temp/input_data_0.pth

    # 모델 또는 데이터를 로드합니다.
    container = torch.jit.load(file_path)

    print(f"/workspace/auto_project/src/cpp/runtime/build/testset_03_pth/temp/input_data_{frame_count % 39}.pth")

    # 로드된 모듈에서 입력 이미지와 CAN 데이터 텐서를 가져옵니다.
    input = container.imgs  # 이미지 데이터 텐서

    # 텐서 차원을 변경합니다.
    input = input.permute(0, 2, 3, 1)

    # 텐서를 float 타입으로 변환합니다.
    input = input.float()

    # 텐서를 메모리에 연속적으로 저장합니다.
    input = input.contiguous()

    return input

# 기존 좌표를 새로운 이미지 크기에 맞게 조정
def scale_coordinates(coords, original_size =(900,1600), new_size = (256,704)):
    return np.round(coords * np.array(new_size) / np.array(original_size)).astype(int)

# 좌표 클리핑: 이미지 범위를 넘어서는 좌표를 클리핑합니다.
def clip_coordinates(coords, new_size =(256,704)):
    return np.clip(coords, 0, np.array(new_size) - 1)

def resize_center_crop(img, new_size):
    original_size = img.shape[1], img.shape[0]
    new_width, new_height = new_size
    
    center_x, center_y = original_size[0] // 2, original_size[1] // 2
    
    crop_x1 = max(center_x - new_width // 2, 0)
    crop_x2 = min(center_x + new_width // 2, original_size[0])
    crop_y1 = max(center_y - new_height // 2, 0)
    crop_y2 = min(center_y + new_height // 2, original_size[1])
    
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    resized_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_img

def resize_center_expand(img, target_size):
    original_h, original_w = img.shape[:2]
    target_w, target_h = target_size

    # Calculate padding for width and height to maintain center
    pad_w = (target_w - original_w) // 2
    pad_h = (target_h - original_h) // 2

    # Add padding around the original image to match the target size
    expanded_img = cv2.copyMakeBorder(
        img, 
        pad_h, pad_h, pad_w, pad_w, 
        borderType=cv2.BORDER_CONSTANT, 
        value=(0, 0, 0)  # Black padding
    )

    # If the target size is not an exact match, further resize to the exact size
    expanded_img = cv2.resize(expanded_img, (target_w, target_h))

    return expanded_img

def visualize_acc_moon1(args, sample_token, nusc_submissions, infos ,img_idx):

    vis_dir = Path(args.save_path)
    vis_dir.mkdir(parents=True, exist_ok=True)
    # scale_factor = args.scale_factor
    scale_factor = 1
    canva_size = args.canva_size
    show_range = args.show_range

    pred_res = nusc_submissions["results"][infos["token"]]

    pred_boxes = [
        pred_res[rid]["translation"]
        + pred_res[rid]["size"]
        + [Quaternion(pred_res[rid]["rotation"]).yaw_pitch_roll[0] + np.pi / 2]
        for rid in range(len(pred_res))
    ]
    
    if len(pred_boxes) == 0:
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
    else:
        pred_boxes = np.array(pred_boxes, dtype=np.float32)
        boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
        corners_global = boxes.corners.numpy().reshape(-1, 3)
        corners_global = np.concatenate(
            [corners_global, np.ones([corners_global.shape[0], 1])], axis=1
        )
        l2g = get_lidar2global(infos)
        corners_lidar = corners_global @ np.linalg.inv(l2g).T
        corners_lidar = corners_lidar[:, :3]
    pred_flag = np.ones((corners_lidar.shape[0] // 8,), dtype=bool)
    scores = [pred_res[rid]["detection_score"] for rid in range(len(pred_res))]
    if args.draw_gt:
        gt_boxes = infos["gt_boxes"]
        gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
        width = gt_boxes[:, 4].copy()
        gt_boxes[:, 4] = gt_boxes[:, 3]
        gt_boxes[:, 3] = width
        corners_lidar_gt = (
            LB(infos["gt_boxes"], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        )
        corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)
        gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
        pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)
        scores = scores + [0 for _ in range(infos["gt_boxes"].shape[0])]
    scores = np.array(scores, dtype=np.float32)
    sort_ids = np.argsort(scores)

    # image view
    imgs = []
    input_imgs = torch_jit_load(img_idx)
    # for view in views:
    for idx, view in enumerate(views):
        # print ("image path:", infos["cams"][view]["data_path"])
        # img = cv2.imread(infos["cams"][view]["data_path"])
        img = input_imgs[idx].numpy()
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)

        # img = cv2.resize(img, (704, 256))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# RGB -> BGR 변환
        # img = resize_center_crop(img, (1600, 900))
        # img = cv2.resize(img, (1600, 900))
        img = resize_center_expand(img,(1600,900))

        # draw instances
        corners_img, valid = lidar2img(corners_lidar, infos["cams"][view])
        valid = np.logical_and(
            valid, check_point_in_img(corners_img, img.shape[0], img.shape[1])
            # valid, check_point_in_img(corners_img, 900, 1600)
        )
        # corners_img = scale_coordinates(corners_img)
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(int)
        
        for aid in range(valid.shape[0]):
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(
                        img,
                        corners_img[aid, index[0]],
                        corners_img[aid, index[1]],
                        color=color_map[int(pred_flag[aid])],
                        thickness=scale_factor,
                    )
        # img = cv2.resize(img, (704, 256))
        # img = resize_center_crop(img, (704, 256))
        imgs.append(img)

    # bird-eye-view
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
    
    # draw instances (중간canvas에 bev instance drawing)
    corners_lidar = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = (
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    )
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)

    # # 조정된 좌표 계산
    # bottom_corners_bev = scale_coordinates(bottom_corners_bev)
    # center_canvas = scale_coordinates(center_canvas)
    # head_canvas = scale_coordinates(head_canvas)

    # # 좌표 클리핑: 이미지 범위를 넘어서는 좌표를 클리핑합니다.
    # bottom_corners_bev = clip_coordinates(bottom_corners_bev)
    # center_canvas = clip_coordinates(center_canvas)
    # head_canvas = clip_coordinates(head_canvas)

    for rid in sort_ids:
        score = scores[rid]
        if score < args.vis_thred and pred_flag[rid]:
            continue
        score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
        color = color_map[int(pred_flag[rid])]

        color = [color[2], color[1], color[0]]  # RGB -> BGR 변환
        color = np.clip(np.array(color) * score, 0, 255).astype(int)  # 점수 반영 및 클리핑 
        
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                [color[0] * score, color[1] * score, color[2] * score],
                thickness=1,
            )
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            [color[0] * score, color[1] * score, color[2] * score],
            1,
            lineType=8,
        )

    # Img height = 256 / Img width = 704 반영
    # fuse image-view and bev
    img_height = 900
    img_width = 1600
    img = np.zeros((img_height * 2 + canva_size * scale_factor, img_width * 3, 3), dtype=np.uint8)
    img[:img_height, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate(
        [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1
    )
    img[img_height + canva_size * scale_factor :, :, :] = img_back
    img = cv2.resize(
        img,
        (int(img_width / scale_factor * 3), int(img_height / scale_factor * 2 + canva_size)),
    )
    w_begin = int((img_width * 3 / scale_factor - canva_size) // 2)
    img[
        int(img_height / scale_factor) : int(img_height / scale_factor) + canva_size,
        w_begin : w_begin + canva_size,
        :,
    ] = canvas

    return img

########### npu 에서 bev heat map feature 만 불러온것 display ###########
def visualize_acc_moon(args, sample_token, nusc_submissions, infos):
    vis_dir = Path(args.save_path)
    vis_dir.mkdir(parents=True, exist_ok=True)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    pred_res = nusc_submissions["results"][infos["token"]]

    pred_boxes = [
        pred_res[rid]["translation"]
        + pred_res[rid]["size"]
        + [Quaternion(pred_res[rid]["rotation"]).yaw_pitch_roll[0] + np.pi / 2]
        for rid in range(len(pred_res))
    ]
    
    if len(pred_boxes) == 0:
        corners_lidar = np.zeros((0, 3), dtype=np.float32)
    else:
        pred_boxes = np.array(pred_boxes, dtype=np.float32)
        boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
        corners_global = boxes.corners.numpy().reshape(-1, 3)
        corners_global = np.concatenate(
            [corners_global, np.ones([corners_global.shape[0], 1])], axis=1
        )
        l2g = get_lidar2global(infos)
        corners_lidar = corners_global @ np.linalg.inv(l2g).T
        corners_lidar = corners_lidar[:, :3]
        
    pred_flag = np.ones((corners_lidar.shape[0] // 8,), dtype=bool)
    scores = [pred_res[rid]["detection_score"] for rid in range(len(pred_res))]
    
    if args.draw_gt:
        gt_boxes = infos["gt_boxes"]
        gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
        width = gt_boxes[:, 4].copy()
        gt_boxes[:, 4] = gt_boxes[:, 3]
        gt_boxes[:, 3] = width
        corners_lidar_gt = (
            LB(infos["gt_boxes"], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        )
        corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)
        gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
        pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)
        scores = scores + [0 for _ in range(infos["gt_boxes"].shape[0])]
        
    scores = np.array(scores, dtype=np.float32)
    sort_ids = np.argsort(scores)

    # bird-eye-view
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
    
    # draw instances
    corners_lidar = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev = (
        (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    )
    bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
    center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas = canter_canvas.astype(np.int32)
    head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas = head_canvas.astype(np.int32)

    for rid in sort_ids:
        score = scores[rid]
        if score < args.vis_thred and pred_flag[rid]:
            continue
        score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
        color = color_map[int(pred_flag[rid])]
        for index in draw_boxes_indexes_bev:
            cv2.line(
                canvas,
                bottom_corners_bev[rid, index[0]],
                bottom_corners_bev[rid, index[1]],
                [color[0] * score, color[1] * score, color[2] * score],
                thickness=1,
            )
        cv2.line(
            canvas,
            center_canvas[rid],
            head_canvas[rid],
            [color[0] * score, color[1] * score, color[2] * score],
            1,
            lineType=8,
        )

    # BBox가 그려진 캔버스를 반환
    return canvas