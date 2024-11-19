import os
os.environ["SAPEONRT_EXTRACT_COMMANDS"] = "1"
from utils import *
import numpy as np 
import onnxruntime as ort
import glob
import os
from threading import Thread
import pathlib

output_dir = "bevdet"
onnx_file_path = "bevdet_onnx"
chipset = "x330"
data_type = "nf8"
batch_size = 1
onnx_lists = glob.glob(f"{onnx_file_path}/**/*.onnx", recursive=True)


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
threads = []

output_path_list = []
for onnx_file in onnx_lists:
    file_name = pathlib.Path(onnx_file).stem

    output_path = f"{output_dir}/{file_name}"
    if file_name.find("simp") == -1:
        continue
    print(file_name)
    if file_name.find("imgnet_backbone") >= 0 :
        batch_size = 6
    else:
        batch_size = 1
    output_path_list.append(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    worker = Thread(target=run_compile, args=(onnx_file, batch_size, chipset, data_type, output_path))
    worker.start()
    threads.append(worker)

for worker in threads:
    worker.join()

chipset = "x330"
model_lists = [f"{f}/result.smp" for f in output_path_list]

model_lists = [
    "bevdet/imgnet_backbone_neck_simp/result.smp",
    "bevdet/imgnet_depthnet_0_simp/result.smp",
    "bevdet/imgnet_depthnet_1_simp/result.smp",
    "bevdet/imgnet_depthnet_2_simp/result.smp",
    "bevdet/imgnet_depthnet_3_simp/result.smp",
    "bevdet/imgnet_depthnet_4_simp/result.smp",
    "bevdet/imgnet_depthnet_5_simp/result.smp",
    "bevdet/bev_encoder_preprocess_simp/result.smp",
    "bevdet/bev_encoder_simp/result.smp",
]
core_list = [1, 2, 3, 4, 2, 3, 4, 1, 1]
runtimes = [get_sapeon_runtime(chipset, model, core_list[idx%len(core_list)], 1) for idx, model in enumerate(model_lists)]
model_info = runtimes[-1].GetModelInfo()

model_configs = [generate_inputs(runtime) for runtime in runtimes]
inputs = list(map(lambda t: t[0], model_configs))
options = list(map(lambda t: t[1], model_configs))
results = run_inference(runtimes, inputs, options, iters = 1, return_outputs = False)
arrange_cmd_dumps(model_lists, "bevdet_cmd_x330_new_with_dummy")