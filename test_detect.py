import cv2
import platform
from PIL import Image
import face_recognition
import face_detect_xavier
from queue import Queue
from multiprocessing import Process
import multiprocessing
import time
from time import sleep
import math
from sklearn import neighbors
import os
import os.path
import pickle
from http import server
import socketserver
import logging
import numpy as np
from datetime import datetime, timedelta

USE_EDGETPU = 1
USE_JETSONINFER= 2
USE_DLIB = 3
FACE_DETECTOR = USE_JETSONINFER
EDGETPU_CROP_FACTORS = [0.5, 1]

SOURCE_IPCAMERA = 2
SOURCE_FILE = 1
SOURCE_USBCAMERA = 0
VIDEO_SOURCE = SOURCE_IPCAMERA

VIDEO_LOG_FNAME = 'mm'

VIDEO_SOURCE_FNAME = "/xavier_ssd/visi/mainUSB.avi"
#VIDEO_SOURCE_TYPE_DEMUX = '' # '' - for *.h264 or VIDEO_SOURCE_TYPE_DEMUX = 'avidemux !' for *.avi
VIDEO_SOURCE_TYPE_DEMUX = 'avidemux !' # '' - for *.h264 or VIDEO_SOURCE_TYPE_DEMUX = 'avidemux !' for *.avi

IPCamera = {'IP':'192.168.0.200', 'pass': '12345', 'login': 'admin'}

HTML_PAGE="""\
<html>
<head>
<title>Face recognition</title>
</head>
<body>
<center><h1>Cam</h1></center>
<center><img src="stream.mjpg" width="1280" height="720" /></center>
</body>
</html>
"""

def camThread(frameBuffer, results, MJPEGQueue, persBuffer, stop_prog):
    pipelines=[]
    FPS_SOURCE = []
    RESOLUTION_SOURCE = []
    # ----------------- GSTREAMER pipeline for USB-webcam ----------------- #
    gstreamer_pipeline = (
    'v4l2src device=/dev/video{cam} do-timestamp=true ! video/x-raw, width={width}, height={height}, framerate=30/1 ! videoflip method=4 !'
    'tee name=streams '
    'streams. ! queue ! videoconvert ! video/x-raw, format=BGR ! appsink '
    'streams. ! videoconvert ! omxh264enc bitrate=8000000 preset-level=2 profile=2 control-rate=1 ! '
    'video/x-h264 ! avimux ! queue ! filesink location=/xavier_ssd/LogVideo_{cam}.avi'
    ).format(cam=0, width=1280, height=720)
    FPS_SOURCE.append(30)
    RESOLUTION_SOURCE.append((1280, 720))
    pipelines.append(gstreamer_pipeline)
    # ----------------- GSTREAMER pipeline for USB-webcam ----------------- #

    # ----------------- GSTREAMER pipeline for video file ----------------- #
    gstreamer_pipeline = (
    'filesrc location={filepath} ! {demux} queue size = 100 ! h264parse ! omxh264dec ! queue size=100 ! '
    'nvvidconv ! video/x-raw,format=BGRx, width=1280, height=720 ! videorate ! video/x-raw, framerate=15/1 ! queue size=100 ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'
    ).format(filepath=VIDEO_SOURCE_FNAME, demux=VIDEO_SOURCE_TYPE_DEMUX)
    FPS_SOURCE.append(15)
    RESOLUTION_SOURCE.append((1280, 720))
    pipelines.append(gstreamer_pipeline)
    # ----------------- GSTREAMER pipeline for video file ----------------- #

    # ----------------- GSTREAMER pipeline for RTSP IP camera ----------------- #
    gstreamer_pipeline = (
        'rtspsrc location=rtsp://{login}:{password}@{cameraIP}/mpeg4 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=BGRx, width=1280, height=720 ! videoconvert ! video/x-raw, format=BGR ! appsink'
    ).format(cameraIP=IPCamera['IP'], login=IPCamera['login'], password=IPCamera['pass'])
    FPS_SOURCE.append(25)
    RESOLUTION_SOURCE.append((1280, 720))
    pipelines.append(gstreamer_pipeline)
    # ----------------- GSTREAMER pipeline for RTSP IP camera ----------------- #
    
    # ----------------- GSTREAMER pipeline for logging to file ----------------- #
    gstreamerpipelinewriter = (
        'appsrc ! queue ! videoconvert ! video/x-raw ! omxh264enc bitrate=8000000 preset-level=2 profile=2 control-rate=1 ! video/x-h264 ! avimux ! queue ! '
        'filesink location=/xavier_ssd/Log_Video_{log}.avi'
    ).format(log=VIDEO_LOG_FNAME)
    # ----------------- GSTREAMER pipeline for logging to file ----------------- #

    gstreamer_pipeline = pipelines[VIDEO_SOURCE]
    print(gstreamer_pipeline[0])
    cam = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    vidlogger = cv2.VideoWriter(gstreamerpipelinewriter, 0, FPS_SOURCE[VIDEO_SOURCE], RESOLUTION_SOURCE[VIDEO_SOURCE])
    # ----------------- Settings for USB-webcam ----------------- #
    #cam.set(cv2.CAP_PROP_FPS, 15)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # ----------------- Settings for USB-webcam ----------------- #
    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    t0 = time.monotonic()
    last_result=None
    frames_cnt = 0
    persons = None
    LUT = np.empty((1,256), np.uint8)
    for i in range(256):
        LUT[0,i] = np.clip(pow(i/256.0, 0.6)*255.0, 0, 255)
    while cam:
        ret, frame = cam.read()
        frame = cv2.LUT(frame, LUT)
        #for c in range(0, 2):
        #    frame[:,:,c] = cv2.equalizeHist(frame[:,:,c])
        if not ret:
            continue
        if frameBuffer.empty():
            frameBuffer.put(frame.copy())
        res = None
        if not results.empty():
            res = results.get(False)
            imdraw = overlay_on_image(frame, res)
            last_result = res
        else:
            imdraw = overlay_on_image(frame,last_result)
        if not persBuffer.empty():
            persons = persBuffer.get(False)
        cv2.imshow('Video', imdraw)
        vidlogger.write(imdraw)
        frames_cnt += 1
        if frames_cnt >= 15:
            t1 = time.monotonic()
            print('FPS={d:.1f}'.format(d = frames_cnt/(t1-t0)))
            frames_cnt = 0
            t0 = t1
        if not MJPEGQueue.full():
            MJPEGQueue.put(imdraw.copy())
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            stop_prog.set()
            break
    stop_prog.set()
    cam.release()
    vidlogger.release()

def overlay_on_image(frame, result):
    if isinstance(result, type(None)):
        return frame
    img = frame
    boxes = result["boxes"]
    encod = result["names"]
    cols = result["color"]
    for box, name, col in zip(boxes,encod,cols):
        y0, x1, y1, x0 = box
        cv2.rectangle(img, (x0,y0), (x1,y1), col, 3)
    return img

def recognition(frameBuffer, objsBuffer, persBuffer, stop_prog):
    if FACE_DETECTOR == USE_EDGETPU:
        model_path = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
        interpreter = face_detect_xavier.initDetector(model_path)
    elif FACE_DETECTOR == USE_JETSONINFER:
        net1 = face_detect_xavier.initJetsonFaceNet()
    net = face_detect_xavier.initJetsonPedNet()
    with open("trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)
    known_persons={}
    for class_dir in os.listdir("faces_base/"):
        face_image = cv2.imread(os.path.join("faces_base/", class_dir, "1.jpg"))
        face_image = cv2.resize(face_image, (288,216))
        known_persons[class_dir]={
            "first_seen": datetime(1,1,1),
            "name": class_dir,
            "first_seen_this_interaction": datetime(1,1,1),
            "last_seen": datetime(1,1,1),
            "seen_count": 0,
            "seen_frames": 0,
            "face_image": face_image
        }
    tfps = time.monotonic()
    objs=[]
    if FACE_DETECTOR == USE_JETSONINFER or FACE_DETECTOR == USE_EDGETPU:
        objs1=[]
    while True:
        if stop_prog.is_set():
            break
        if frameBuffer.empty():
            continue
        t0 = time.monotonic()
        bgr_img = frameBuffer.get()
        rgb_img = bgr_img[:, :, ::-1].copy()
        if FACE_DETECTOR == USE_EDGETPU:
            arr_img = Image.fromarray(rgb_img)
            objs1 = face_detect_xavier.detect_face_HiRes(arr_img, EDGETPU_CROP_FACTORS, interpreter)
        elif FACE_DETECTOR == USE_JETSONINFER:
            objs1 = face_detect_xavier.jetson_infer(rgb_img, net1, 0.6)
        objs = face_detect_xavier.jetson_infer(rgb_img, net, 0.6)
        
        coral_boxes = []
        if FACE_DETECTOR == USE_JETSONINFER or FACE_DETECTOR == USE_EDGETPU:
            for obj in objs1:
                y0 = obj[0]
                x1 = obj[1]
                y1 = obj[2]
                x0 = obj[3]
                coral_boxes.append((y0, x1, y1, x0))
        else:
            coral_boxes=face_recognition.face_locations(rgb_img, model='cnn')
        locR = []
        predR = []
        Color = []
        i = 1
        for loc in objs:
            y0 = int(loc[0])
            x1 = int(loc[1])
            y1 = int(loc[2])
            x0 = int(loc[3])
            locR.append((y0, x1, y1, x0))
            predR.append('{}'.format(i))
            Color.append((255,0,0))
            i +=1
        if coral_boxes:
            enc = face_recognition.face_encodings(rgb_img, known_face_locations=coral_boxes)
            closest_distances = knn_clf.kneighbors(enc, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.55 for i in range(len(coral_boxes))]
            for pred, loc, rec in zip(knn_clf.predict(enc), coral_boxes, are_matches):
                if rec:
                    person_found = known_persons.get(pred)
                    if person_found != None:
                        if known_persons[pred]["first_seen"] != datetime(1,1,1):
                            known_persons[pred]["last_seen"] = datetime.now()
                            known_persons[pred]["seen_frames"] += 1
                            if datetime.now() - known_persons[pred]["first_seen_this_interaction"] > timedelta(minutes=5):
                                known_persons[pred]["first_seen_this_interaction"] = datetime.now()
                                known_persons[pred]["seen_count"] += 1
                                known_persons[pred]["seen_frames"] = 0
                        else:
                            known_persons[pred]["first_seen"] = datetime.now()
                            known_persons[pred]["last_seen"] = datetime.now()
                            known_persons[pred]["seen_count"] += 1
                            known_persons[pred]["first_seen_this_interaction"] = datetime.now()
                    predR.append(pred)
                    Color.append((0,255,0))
                else:
                    predR.append("unknown")
                    Color.append((0,0,255))
                locR.append((int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])))
        if locR:
            if objsBuffer.empty():
                objsBuffer.put({"boxes": locR, "names": predR, 'color': Color})
        else:
            if objsBuffer.empty():
                objsBuffer.put(None)
        te = time.monotonic()-t0
        print('Detect cycle: {0:.1f}msec, FPS: {1:0.1f}fps'.format(te*1000, 1/(te+t0-tfps)))
        tfps = time.monotonic()
        dtnow = datetime.now()
        visi_faces = []
        for pers in known_persons:
            if dtnow-known_persons[pers]["last_seen"] < timedelta(seconds=0.1) and known_persons[pers]["seen_frames"] > 1:
                visi_faces.append(known_persons[pers])
        if persBuffer.empty():
            if len(visi_faces) > 0:
                persBuffer.put(visi_faces)
            else:
                persBuffer.put(None)
        del bgr_img
        del rgb_img
        del dtnow

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            stri = HTML_PAGE
            content = stri.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Conent-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        # elif self.path == '/data.html':
        #     stri = coral_engine.result_str
        #     content = stri.encode('utf-8')
        #     self.send_response(200)
        #     self.send_header('Content-Type', 'text/html')
        #     self.send_header('Conent-Length', len(content))
        #     self.end_headers()
        #     self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if not self.server.MJPEGQueue.empty():
                        frame = self.server.MJPEGQueue.get()
                        ret, buf = cv2.imencode('.jpg', frame)
                        frame = np.array(buf).tostring()
                        self.wfile.write(b'-FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\r')
            except Exception as e:
                logging.warning('Removed streaming clients %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def server_start(frameQueue, exit_key):
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.MJPEGQueue = frameQueue
        print('Started server')
        server.serve_forever()
    finally:
        exit_key.set()

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    prog_stop = multiprocessing.Event()
    prog_stop.clear()
    prog_stop1 = multiprocessing.Event()
    prog_stop1.clear()
    recImage = multiprocessing.Queue(2)
    resultRecogn = multiprocessing.Queue(2)
    persBuffer = multiprocessing.Queue(2)
    MJPEGQueue = multiprocessing.Queue(10)
    camProc = Process(target=camThread, args=(recImage, resultRecogn, MJPEGQueue, persBuffer,  prog_stop), daemon=True)
    camProc.start()
    frecogn = Process(target=recognition, args=(recImage, resultRecogn, persBuffer, prog_stop), daemon=True)
    frecogn.start()
    serverProc = Process(target=server_start, args=(MJPEGQueue, prog_stop1), daemon=True)
    serverProc.start()

    while True:
        if prog_stop.is_set():
            camProc.terminate()
            frecogn.terminate()
            serverProc.terminate()
            break
        sleep(1)
    cv2.destroyAllWindows()
