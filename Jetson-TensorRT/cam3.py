from tensorflow.python.saved_model import tag_constants
from jetcam.csi_camera import CSICamera
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import cv2
import time
from threading import Thread
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import tensorflow as tf
import numpy as np
from datetime import datetime

#model = tf.keras.models.load_model('mobilenet_v3_large_au_224_finet_model.h5', compile=False)
labels = ["clear", "dirty", "obstruction", "shadow"]

q = Queue(900)

#gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP32)
SAVED_MODEL_DIR = "myModel"
FP32_SAVED_MODEL_DIR = SAVED_MODEL_DIR+"_TFTRT_FP32/1"

saved_model_loaded = tf.saved_model.load(
    SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

elapsedTime = 0
global start
global countFrames
start = None
countFrames = 0

#videoRecorder = cv2.VideoWriter(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 3, (1920, 1080))


def put_in_queue(change):
    frame = change['new']
    #start_time = time.time()
    #pred = np.argmax(model.predict(input_tensor), axis=-1)
    # print(labels[pred[0]])
    #q.put((frame, labels[pred[0]]))
    q.put(frame)
    global countFrames
    countFrames += 1
    #start_time = time.time()
    print("Q size in: " + str(q.qsize()))


def get_from_queue():
    count = 0
    countM = 0
    videoRecorder = cv2.VideoWriter(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 3, (1920, 1080))
    while countM < 5:
        while q.empty():
            time.sleep(.01)
        if count < 150:
            videoRecorder.release()
            print("finished video")
            count = 0
            countM += 1
        if countM > 1 and count == 0:
            videoRecorder = cv2.VideoWriter(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 3, (1920, 1080))
	#frame, label = q.get()
        if q.qsize() > 30:
            predArray = []
            imgs = []
            while len(predArray) != 30:
                img = q.get()
                imgs.append(img)
                image = cv2.resize(img, (224, 224))
                image = np.asarray(image)
                input_tensor = tf.convert_to_tensor(image)
                input_tensor = input_tensor[tf.newaxis, ...]
                predArray.append(input_tensor)
            #print(f'len of array: {len(predArray)}')
            global camera
            camera.unobserve(put_in_queue, names='value')
            # print(predArray)
            start_time = time.time()
            #preds = np.argmax(model.predict(np.vstack(predArray)), axis=-1)
            #preds = model.predict(np.vstack(predArray))
            preds = infer(predArray)
            #print(f"elapsed time for predict {time.time()-start_time}")
            print("elapsed time for predict")
            print(str(time.time()-start_time))
            i = 0
            for pred in preds:
                frame = imgs[i]
                pred = np.argmax(pred, axis=-1)
                label = labels[pred]
                print(label)
                cv2.putText(frame, label, (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 2)
                videoRecorder.write(frame)
                i += 1
            predArray = []

        # print(label)
        #end_time = time.time()
        #total_time = end_time - start_time
        #print(f"elapsed time for processing {total_time}")
        #print("Q size out: "+ str(q.qsize()))
        #count +=1
        #cv2.putText(frame, label, (20,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 2)
        # videoRecorder.write(frame)
    #global camera
    #camera.unobserve(put_in_queue, names='value')


global camera
camera = CSICamera(capture_width=1920, capture_height=1080, width=1920,
                   height=1080, capture_device=0)  # confirm the capture_device number

image = camera.read()

print(image.shape)

print(camera.value.shape)

image = cv2.resize(image, (224, 224))
image = np.asarray(image)
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]
#start_time = time.time()
pred = np.argmax(model.predict(input_tensor), axis=-1)
print(labels[pred[0]])

cv2.imwrite("test.png", image)
camera.running = True
start = datetime.now()
camera.observe(put_in_queue, names='value')

tvr = Thread(target=get_from_queue)
tvr.start()
# tvr.join()

print("finished thread")

#camera.unobserve(put_in_queue, names='value')
#end = datetime.now()
print("finished camera")

#elapsedTime = (end-start).total_seconds()
# print(elapsedTime)
# print(countFrames/elapsedTime)
