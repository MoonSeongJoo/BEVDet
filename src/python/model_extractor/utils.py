import cv2
import numpy as np
import onnx
import onnxruntime as rt
import matplotlib.pyplot as plt
import sapeon.runtime as SapeonRT
import time

def run_calibration(model_file_name, image_data, calib_file_name = "calib.txt"):

  EP_list = ['CPUExecutionProvider']
  provider_options = [{}]

  session_options = rt.SessionOptions()
  rt.set_default_logger_severity(0)

  model = onnx.load(model_file_name)

  ort_session = rt.InferenceSession(model.SerializeToString(),
      session_options, providers=EP_list, provider_options = provider_options)
  org_output_names = [x.name for x in ort_session.get_outputs()]

  # add all intermediate outputs to onnx net
  for node in model.graph.node:
    for output in node.output:
      if output not in org_output_names:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])

  ort_session = rt.InferenceSession(model.SerializeToString(),
      session_options, providers=EP_list, provider_options = provider_options)

  input_name = ort_session.get_inputs()[0].name

  batch, channel, height, width = list(ort_session.get_inputs())[0].shape
  if type(batch) != int:
    batch = 1

  if len(image_data.shape) == 3:
    image_data = np.expand_dims(image_data,0)

  image_data = image_data.astype(np.float32)
  input_thres = np.quantile(abs(image_data), 0.999) # 99.9% percentile
  
  all_input_names = [x.name for x in ort_session.get_inputs()]
  all_output_names = [x.name for x in ort_session.get_outputs()]
  all_output_values = ort_session.run(all_output_names, {input_name: image_data})
  all_output_dict = dict(zip(all_output_names, all_output_values))

  quant_file = open(calib_file_name, "w")

  for name in all_input_names:
    fmt = "%s\t%f\n" % (name, input_thres)
    quant_file.write(fmt)

  for name, tensor in all_output_dict.items():
    #thresh = max(abs(tensor.max()), abs(tensor.min())) # Max
    thresh = np.quantile(abs(tensor), 0.999) # 99.9% percentile
    fmt = "%s\t%f\n" % (name, thresh)
    quant_file.write(fmt)
  quant_file.close()

import subprocess

def compile(spear_input, output_path="./binary", mode="fast", chipset="x220", data_type="", ps_file_path="", nr=None):
  cmd = ['snc', '-i', spear_input, '-o', output_path, '-m', mode, '-c', chipset, "--dump_manual_ps"]

  if data_type != '':
    cmd.extend(['-t', data_type])

  if ps_file_path != '':
    cmd.extend(['-p', ps_file_path])
  
  if nr is not None:
    cmd.extend(['-nr', str(nr)])
  
  subprocess.run(cmd, check=True)

def onnx2sapeon(onnx_path, calib_path, spear_path, skip="", sp_layer_threshold=0, input_batch=0, device_type="x220"):
  cmd = ['onnx2sapeon', '--input', onnx_path, '--calib', calib_path, '--output_dir', spear_path, "--device_type", device_type]

  if skip != "":
    cmd.extend(['--skip', skip])

  if sp_layer_threshold != 0:
    cmd.extend(['--sp_layer_thresh', str(sp_layer_threshold)])

  if input_batch != 0:
    cmd.extend(['--input_batch', str(input_batch)])


  subprocess.run(cmd, check=True)

def show_seg_map(img, pred, title, input_size):
    h, w  = pred.shape[1:]
    pred = np.argmax(pred,0)
    pred = np.expand_dims(pred, -1)
    pred = np.tile(pred,(1,1,3))
    output = np.ones((h,w,3), dtype=np.uint8)
    for i in range(19):
        color_map = np.ones((h,w,3))
        label = trainId2label[i]
        color_map = color_map * label.color
        color_map = (pred==(i))*color_map
        color_map = color_map.astype(np.uint8)
        output += color_map
    fig, axes = plt.subplots(1,2)
    fig.suptitle(title)
    output = cv2.resize(output, input_size)
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[1].imshow(output)
    axes[1].axis('off')
    plt.savefig(f'{title}.png', dpi=100)

def preprocess(img, dims=None, need_transpose=False):
    output_height, output_width, _ = dims
    img = cv2.resize(img, (output_height,output_width))
    img = np.asarray(img, dtype='float32')
    img = img / 255
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = (img - mean) / std
    img = img.astype(np.float32)
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


# SAPEONRT_EXTRACT_COMMANDS 1
def get_sapeon_runtime(chipset, model_file, use_cores, device_id = 0):
    core_id = [b'0000', b'0001', b'0010', b'0100', b'1000']
    if chipset == "x330":
        runtime = SapeonRT.MakeSapeonRuntime(SapeonRT.SapeonDeviceType.X330, device_id, core_id[use_cores])
    else:
        runtime = SapeonRT.MakeSapeonRuntime(SapeonRT.SapeonDeviceType.X340, device_id, core_id[use_cores])
    runtime.OpenDevice()
    runtime.SetModel(model_file)
    return runtime

def generate_inputs(runtime, input_data=None):
    model_info = runtime.GetModelInfo()
    input_data = []
    for input in model_info.inputs:
      shape = [input.shape[0], input.shape[2], input.shape[1], input.shape[3]]
      data = np.random.rand(*shape).astype(np.float32)
      tensor = SapeonRT.Tensor.from_numpy(data, SapeonRT.Tensor.Format.NHWC)
      input_data.append(SapeonRT.Port(tensor, input.name))
    option = SapeonRT.InferenceOptions()
    option.output_format = SapeonRT.Tensor.Format.NHWC
    return input_data, option

def run_inference(runtimes, inputs, options, iters = 1000, return_outputs = False):
    t_CreateInferenceContext = [0]*len(runtimes)
    t_ExecuteGraph = [0]*len(runtimes)
    t_WaitInferenceDone = [0]*len(runtimes)
    t_GetResult = [0]*len(runtimes)

    t_CreateInferenceContext_sq = [0]*len(runtimes)
    t_ExecuteGraph_sq = [0]*len(runtimes)
    t_WaitInferenceDone_sq = [0]*len(runtimes)
    t_GetResult_sq = [0]*len(runtimes)
    
    for i in range(iters):
        outputs = []
        for j, (runtime, input_data, option) in enumerate(zip(runtimes, inputs, options)):
            tic = time.time()
            context = runtime.CreateInferenceContext(input_data, option)
            toc = time.time()
            t_CreateInferenceContext[j] += (toc-tic)
            t_CreateInferenceContext_sq[j] += (toc-tic)**2

            tic = time.time()
            runtime.ExecuteGraph(context)
            toc = time.time()
            t_ExecuteGraph[j] += (toc-tic)
            t_ExecuteGraph_sq[j] += (toc-tic)**2

            tic = time.time()
            runtime.WaitInferenceDone(context)
            toc = time.time()
            t_WaitInferenceDone[j] += (toc-tic)
            t_WaitInferenceDone_sq[j] += (toc-tic)**2

            tic = time.time()
            results = runtime.GetResult(context)
            
            outputs.append(results)
            toc = time.time()
            t_GetResult[j] += (toc-tic)
            t_GetResult_sq[j] += (toc-tic)**2

    t_CreateInferenceContext = list(map(lambda t: t / iters, t_CreateInferenceContext))
    t_ExecuteGraph = list(map(lambda t: t / iters, t_ExecuteGraph))
    t_WaitInferenceDone = list(map(lambda t: t / iters, t_WaitInferenceDone))
    t_GetResult = list(map(lambda t: t / iters, t_GetResult))

    t_CreateInferenceContext_sq = list(map(lambda t: t / iters, t_CreateInferenceContext_sq))
    t_ExecuteGraph_sq = list(map(lambda t: t / iters, t_ExecuteGraph_sq))
    t_WaitInferenceDone_sq = list(map(lambda t: t / iters, t_WaitInferenceDone_sq))
    t_GetResult_sq = list(map(lambda t: t / iters, t_GetResult_sq))

    t_CreateInferenceContext_sq = list(map(lambda t: t[1] - t[0]**2, zip(t_CreateInferenceContext, t_CreateInferenceContext_sq)))
    t_ExecuteGraph_sq = list(map(lambda t: t[1] - t[0]**2, zip(t_ExecuteGraph, t_ExecuteGraph_sq)))
    t_WaitInferenceDone_sq = list(map(lambda t: t[1] - t[0]**2, zip(t_WaitInferenceDone, t_WaitInferenceDone_sq)))
    t_GetResult_sq = list(map(lambda t: t[1] - t[0]**2, zip(t_GetResult, t_GetResult_sq)))

    if return_outputs == False:
        return t_CreateInferenceContext, t_ExecuteGraph, t_WaitInferenceDone, t_GetResult, \
            t_CreateInferenceContext_sq, t_ExecuteGraph_sq, t_WaitInferenceDone_sq, t_GetResult_sq
    else:
        return outputs
    
def run_compile(model_file, batch_size, chipset, data_type, output_path="./", nr=None):
    onnx2sapeon(onnx_path=model_file,
        calib_path="dummy",
        spear_path=output_path,
        device_type = chipset,
        input_batch=batch_size)

    # spear -> smp
    compile(spear_input=f"{output_path}/spear_1-1.sp",
            output_path=output_path, 
            mode="fast",
            chipset=chipset,
            data_type=data_type,
            nr = nr)
  

import os  
import shutil
import pathlib
import glob

def arrange_cmd_dumps(model_lists, output_dir):
    if pathlib.Path(output_dir).exists():
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd_model_lists = []
    for model in model_lists:
        model_name = model.split("/")[-2]
        model_dir = os.path.join(output_dir, model_name)
        cmd_model_lists.append(model_dir)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    cmd_lists = glob.glob("SAPEON_CMD_DUMP_*")
    cmd_lists.sort(key= lambda x: int(x.split("_")[3]))
    
    # copy cps and weights
    for idx, model in enumerate(model_lists):
        cps_file_path = cmd_lists[idx*2]
        weight_file_path = cmd_lists[idx*2+1]
        shutil.move(cps_file_path, cmd_model_lists[idx])
        shutil.move(weight_file_path, cmd_model_lists[idx])

    # copy input icvt inference ocvt
    idx = len(model_lists)*2
    model_idx = -1

    for idx in range(idx, len(cmd_lists)):
        cmd_file = cmd_lists[idx]
        if cmd_file.find("dma_write") != -1:
            model_idx += 1
        shutil.move(cmd_file, cmd_model_lists[model_idx])