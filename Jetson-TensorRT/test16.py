from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import random
import shutil
import pathlib
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import time
import numpy as np
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
SAVED_MODEL_DIR = "mobilenetv3large"
FP16_SAVED_MODEL_DIR = SAVED_MODEL_DIR+"_TFTRT_FP16/1"
batch_size = 32
img_height = 224
img_width = 224

IMG_SIZE = (img_height, img_width)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path("test"),
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
    )
class_names = test_ds.class_names
print(class_names)
# load saved model
saved_model_loaded = tf.saved_model.load(FP16_SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

print('Warming up for 50 batches...')
cnt = 0
for x, y in test_ds:
    labeling = infer(x)
    cnt += 1
    if cnt == 50:
        break

print('Benchmarking inference engine...')
num_hits = 0
num_predict = 0
start_time = None
for x, y in test_ds:
    labeling = infer(x)
    if start_time == None:
        start_time = time.time()
    preds = np.around(labeling['dense'].numpy())
    hits = np.logical_and(preds, y.numpy())
    hits_count = np.count_nonzero(hits)
    num_hits += hits_count
    num_predict += preds.shape[0]

print('Accuracy: %.2f%%'%(100*num_hits/num_predict))
print('Inference speed: %.2f samples/s'%(num_predict/(time.time()-start_time)))
