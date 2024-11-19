
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import uvicorn
import threading
import time
import zmq
import xml.etree.ElementTree as ET
from gzip import GzipFile
from io import BytesIO
from fastapi import FastAPI,WebSocket,Request
import glob 
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from det_drawer import BEVDETDrawer
 
mutex = threading.Lock()

app = FastAPI()
bev_drawer = BEVDETDrawer()

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

port = 8000
frame_count = 0
is_updated = False
imgs = None
output_imgs = None
det_result = {}
def http_server():
    global port
    uvicorn.run(app, host="0.0.0.0", port = port)

def tcp_worker():
    context = zmq.Context()
    subscriber = context.socket(zmq.PULL)
    # subscriber.bind(f"tcp://127.0.0.1:5556")
    subscriber.bind(f"tcp://0.0.0.0:5556")
    global frame_count
    global is_updated
    global output_imgs
    global det_result 
    input_imgs = None
    while True:
        multipart = subscriber.recv_multipart()
        msg_type = multipart[0].decode('utf-8')
        try:
            if msg_type =="__start__":
                frame_count = int(multipart[1].decode('utf-8'))
                # if len(det_result) > 0 and input_imgs is not None: # legacy code
                if len(det_result) > 0 : 
                    mutex.acquire()
                    # output_imgs = bev_drawer.get_det_vis(det_result, input_imgs) # legacy code
                    output_imgs = bev_drawer.get_det_vis_moon1(det_result, frame_count)
                    cv2.putText(output_imgs, str(frame_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mutex.release()
                det_result = {}
            elif msg_type == "imgs":
                input_imgs = multipart[1]
                input_imgs = np.frombuffer(input_imgs, dtype=np.float32)
                input_imgs = input_imgs.reshape(6,256,704,3)
                input_imgs = img_norm_cfg["std"] * input_imgs + img_norm_cfg["mean"]
            else:
                det_result[msg_type] = multipart[1]
        except Exception as e:
            print(e)
            print(f"{msg_type} Error: %s" % multipart[0].decode('utf-8'))
            det_result = {}
            mutex.release()

def generate_video():
    global frame_count
    global imgs
    global output_imgs
    frame = [0]*4
    idx = 0
    prev_frame_count = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 품질을 80으로 설정
    while True:
        mutex.acquire()
        if frame_count != prev_frame_count:
            # _, encoded_image = cv2.imencode('.jpg', output_imgs)
            _, encoded_image = cv2.imencode('.jpg', output_imgs ,encode_param)
            frame = encoded_image.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            prev_frame_count = frame_count
        mutex.release()
        time.sleep(0.033)

@app.get('/')
async def video_feed():
    return StreamingResponse(generate_video(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":

    # Runs server and worker in separate processes
    p1 = threading.Thread(target=http_server)
    p1.start()
    time.sleep(1)  # Wait for server to start
    p2 = threading.Thread(target=tcp_worker)
    p2.start()
    p1.join()
    p2.join()