# SAIT BEVDet

## Getting Started
```shell
cd SAIT_BEV
./docker/up.sh --build
./docker/setup.sh
./docker/bash.sh

ln -s /data/nuscenes /bevdet/data/nuscenes # set the dataset root_path

# default dataset root_path is /data/nuscenes
root@$USER:/bevdet# python bevdet/tools/create_data_bevdet.py

root@$USER:/bevdet# python src/python/single_infer_test.py bevdet/configs/bevdet/bevdet4d-r50-depth-cbgs.py --fuse-conv-bn
```

## Dependencies

| Package        | Version  | Source                                         |
|----------------|----------|------------------------------------------------|
| mmengine       | 0.10.3   | https://github.com/open-mmlab/mmengine         |
| mmcv-full      | 1.6.0    | https://github.com/open-mmlab/mmcv             |
| mmcls          | 0.25.0   | https://github.com/open-mmlab/mmclassification |
| mmdet          | 2.28.2   | https://github.com/open-mmlab/mmdetection      |
| mmdet3d        | 1.0.0rc4 | /mmdetection3d                                 |
| mmsegmentation | 0.30.0   | http://github.com/open-mmlab/mmsegmentation    |

## bev_pool_v2 test on cpu

```shell
cd SAIT_BEV/src/cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./bev_pool_v2_test
```

## Image Network Export

```shell
root@$USER:/bevdet# python src/python/export_image_network.py bevdet/configs/bevdet/bevdet4d-r50-depth-cbgs.py --fuse-conv-bn
```