import os
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
#import pytesseract
import pymysql



CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
    

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Detection from Image
# IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars63.png')

imgF = cv2.imread('bb.jpg')
img = cv2.resize(imgF, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)

# ROI
detection_threshold = 0.3
image = image_np_with_detections
scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

width = image.shape[1]
height = image.shape[0]

for idx, box in enumerate(boxes):
    roi = box * [height, width, height, width]
    region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
    cv2.imshow("ROI", region)

    # OCR
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # conf = "--psm 3"
    #
    # # Convert image to grayscale
    # gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    #
    # # Convert image to black and white (using adaptive threshold)
    # adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
    # cv2.imshow('AT', adaptive_threshold)
    #
    # output = pytesseract.image_to_string(adaptive_threshold, config=conf, lang="eng")
    # print('OCR===>',output)
    #
    # # Text filter
    # final = ''.join(e for e in output if e.isalnum())
    # print('Filter===>', final)
    #
    # # DB Check
    # # database connection
    # cardb = pymysql.connect(host="localhost", user="root", passwd="", database="cardata")
    # cursor = cardb.cursor()
    # # some other statements  with the help of cursor
    # query = "SELECT Reg_No FROM reg_plate WHERE Reg_No = %s"
    # number = final
    # results = cursor.execute(query, number)
    # if results > 0:
    #     print('Welcome u can enter')
    # else:
    #     print('Sorry u are not Authorized')
    # cardb.close()

#cv2.imshow("frame", cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
cv2.imshow("frame", image_np_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()









# # Detection and processing from CAM
# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     image_np = np.array(frame)
#
#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#     detections = detect_fn(input_tensor)
#
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections
#
#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
#
#     label_id_offset = 1
#     image_np_with_detections = image_np.copy()
#
#     viz_utils.visualize_boxes_and_labels_on_image_array(
#         image_np_with_detections,
#         detections['detection_boxes'],
#         detections['detection_classes'] + label_id_offset,
#         detections['detection_scores'],
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=5,
#         min_score_thresh=.7,
#         agnostic_mode=False)
#
#     # ROI
#     detection_threshold = 0.7
#     image = image_np_with_detections
#     scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
#     boxes = detections['detection_boxes'][:len(scores)]
#     classes = detections['detection_classes'][:len(scores)]
#
#     width = image.shape[1]
#     height = image.shape[0]
#
#     for idx, box in enumerate(boxes):
#         roi = box*[height, width, height, width]
#         region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
#         cv2.imshow("ROI",region)
#
#         # OCR
#         pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#         conf = "--psm 3"
#         gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
#         output = pytesseract.image_to_string(gray, config=conf, lang="eng")
#         print(output)
#
#     cv2.imshow('Detection', cv2.resize(image_np_with_detections, (800, 600)))
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         break

