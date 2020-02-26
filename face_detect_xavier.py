import jetson.inference
import jetson.utils
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import Bbox
import cv2

#FRAME_SIZE = (1920, 1080)
CROP_SIZE = (448,448)

def calc_offset(orig_size, crop_size):
    return (orig_size[0] / round(orig_size[0]/crop_size[0]+0.5), orig_size[1] / round(orig_size[1]/crop_size[1]+0.5))

def detect_face_HiRes(arr_img, crop_factors, interpreter):
    crop_imgs = []
    FRAME_SIZE = arr_img.size
    for cr in crop_factors:
        crop_size = [int(i*cr) for i in CROP_SIZE]
        CROP_OFFSET = calc_offset(FRAME_SIZE, crop_size)
        for nx in range(round(FRAME_SIZE[0]/crop_size[0]+0.5)):
            for ny in range(round(FRAME_SIZE[1]/crop_size[1]+0.5)):
                x0 = int(CROP_OFFSET[0]*nx)
                x1 = int(x0+crop_size[0])
                y0 = int(CROP_OFFSET[1]*ny)
                y1 = int(y0+crop_size[1])
                arr_crop_img = arr_img.crop((x0,y0,x1,y1))
                crop_imgs.append({'image': arr_crop_img,
                                'coord': (x0,y0, x1, y1),
                                'crop': (FRAME_SIZE[0]/crop_size[0], FRAME_SIZE[1]/crop_size[1])})
    crop_imgs.append({'image': arr_img,
                    'coord': (0,0, arr_img.size[0]-1, arr_img.size[1]-1),
                    'crop': (1, 1)})

    BATCH_SIZE = len(crop_imgs)
    glob_objs = []
    glob_rects = []
    for crop_img in crop_imgs:
        objs, image = detect_face(crop_img['image'], interpreter)
        for obj in objs:
            y0, x1, y1, x0 = obj
            x0 = int(x0 + crop_img['coord'][0])
            y0 = int(y0 + crop_img['coord'][1])
            x1 = int(x1 + crop_img['coord'][0])
            y1 = int(y1 + crop_img['coord'][1])
            w = x1-x0
            h = y1-y0
            glob_rects.append([x0, y0, w, h])

    if crop_factors:
        glob_rects, weights = cv2.groupRectangles(glob_rects, 1, 0.6)
    for rect in glob_rects:
        x0, y0, w, h = rect
        glob_objs.append((y0, x0+w, y0+h, x0))
    return glob_objs

def initJetsonFaceNet():
    net = jetson.inference.detectNet("facenet", threshold=0.3)
    return net

def initJetsonPedNet():
    #net = jetson.inference.detectNet("multiped", threshold=0.7)
    net = jetson.inference.detectNet("pednet", threshold=0.6)
    return net

def initDetector(model_path):  
    interpreter = tflite.Interpreter(model_path,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, size, resize):
  """Copies a resized and properly zero-padded image to the input tensor.

  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(interpreter)
  w, h = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(interpreter)
  tensor.fill(0)  # padding
  _, _, channel = tensor.shape
  tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
  return scale, scale

def jetson_infer(frame, net, score_threshold):
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = jetson.utils.cudaFromNumpy(frame_rgba)
    height, width, chans = frame.shape
    detections = net.Detect(img, width, height)
    bboxes = []
    for det in detections:
        if det.Confidence >= score_threshold:
            bboxes.append((int(det.Top), int(det.Right), int(det.Bottom), int(det.Left)))
    return bboxes

def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)

def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  class_ids = output_tensor(interpreter, 1)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Bbox.Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=Bbox.BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))
  return [make(i) for i in range(count) if scores[i] >= score_threshold]

def detect_face(frame, interpreter):
    scale = set_input(interpreter, frame.size, lambda size: frame.resize(size, Image.NEAREST))
    interpreter.invoke()
    objs = get_output(interpreter, 0.5, scale)
    glob_objs = []
    glob_rects = []
    for obj in objs:
        bbox = obj.bbox
        x0 = int(bbox.xmin)
        y0 = int(bbox.ymin)
        x1 = int(bbox.xmax)
        y1 = int(bbox.ymax)
        w = x1-x0
        h = y1-y0
        glob_objs.append((y0, x0+w, y0+h, x0))
    return (glob_objs, frame)
