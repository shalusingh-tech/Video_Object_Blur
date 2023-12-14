# import libraries
import tensorflow as tf
import numpy as np
#from PIL import Image
import cv2
import matplotlib.pyplot as plt
#import os
#import tensorflow_hub as hub
#import pandas as pd
import keras
import warnings

# coco dataset class labels
coco_classes = {
0: 'unlabeled',
1: 'person',
2: 'bicycle',
3: 'car',
4: 'motorcycle',
5: "airplane",
6: "bus",
7: "train",
8: "truck",
9: "boat",
10:" traffic light",
11: "fire hydrant",
12: "street sign",
13: "stop |sign",
14: "parking meter",
15: "bench",
16: "bird",
17: "cat",
18: "dog",
19: "horse",
20: "sheep",
21: "cow",
22: "elephant",
23:" bear",
24: "zebra",
25: "giraffe",
26: "hat",
27: "backpack",
28: "umbrella",
29: "shoe",
30: "eye glasses",
31: "handbag",
32:" tie",
33: "suitcase",
34:" frisbee",
35: "skis",
36: "snowboard",
37: "sports ball",
38: "kite",
39: "baseball bat",
40: "baseball glove",
41: "skateboard",
42: "surfboard",
43: "tennis racket",
44: "bottle",
45: "plate",
46: "wine glass",
47: "cup",
48: "fork",
49: "knife",
50: "spoon",
51: "bowl",
52: "banana",
53:"apple",
54:"sandwich",
55:" orange",
56: "broccoli",
57: "carrot",
58: "hot dog",
59:' pizza',
60: "donut",
61: 'cake',
62: "chair",
63: "couch",
64: "potted plant",
65: "bed",
66: "mirror",
67: "dining table",
68: "window",
69: "desk",
70: "toilet",
71: "door",
72: "tv",
73:" laptop",
74: "mouse",
75: "remote",
76:" keyboard",
77: "cell phone",
78: "microwave",
79: "oven",
80: "toaster",
81: "sink",
82: "refrigerator",
83: "blender",
84: "book",
85:" clock",
86: "vase",
87: "scissors",
88: "teddy bear",
89: "hair drier",
90: "toothbrush",
}


#========================================================
# function: blur the image
def blur_image(image,coordinates = None):
  img = image.copy() # copy the image to work on new image
  if (coordinates is not None):
    #print('Performing image blur operation...')
    for coord in (coordinates):
      ymin,xmin,ymax,xmax = coord
      #print('Image shape:',img.shape)
      # Extract region of intrest
      Y_min,X_min,Y_max,X_max = int(ymin*img.shape[0]),int(xmin*img.shape[1]),int(ymax*img.shape[0]),int(xmax*img.shape[1])
      #print('Y_min,Y_max',Y_min,Y_max)
      #print('X_min,X_max',X_min,X_max)
      roi = img[Y_min:Y_max,X_min:X_max]
      #show_img(roi,'Original_roi')
      # blur the extracted img using Gausian blur
      try:
        roi = cv2.GaussianBlur(roi)
        #show_img(roi,title='blured roi')
        # replace the original roi with blured_roi
        img[Y_min:Y_max, X_min:X_max] = roi

      except:
        pass

    return img
#==========================================================


#============================================================================
# function: get image info.
def get_info(img):
  print('Image shape:',img.shape)
  print('Image Dtype:',img.dtype)
  print('Image Min. Value:',tf.reduce_min(img))
  print('Image Max. Value:',tf.reduce_max(img))
#============================================================

#=======================================================================
# function: to show the image
def show_img(img,title = ''):
  try:
    img = img.astype('uint8')
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()
  except:
    print('IMAGE NOT SHOWN!')
#==================================================================

#==================================================================
# function: filter detection boxs
def filter_detection(detector_output,select_classes,thr = 0.6):
  detection_boxs = detector_output['detection_boxes']
  detection_class = detector_output['detection_classes']
  detection_scores = detector_output['detection_scores']
 # get the masking to select classes which user choosed
  masked_classes = np.isin(detection_class,select_classes)

  # select only selected classes
  detection_class = detection_class[masked_classes]
  detection_boxs = detection_boxs[masked_classes]
  detection_scores = detection_scores[masked_classes]

  # filter the detection boxses based on threshold
  selected_scores = detection_scores[detection_scores >= thr]
  selected_class = detection_class[detection_scores >= thr]
  selected_boxs = detection_boxs[detection_scores >= thr].numpy()

  return selected_boxs,selected_class,selected_scores
#==============================================================================

#========================================================================
# function: to get info. of detection output
def detection_info(detector_output):
  print('Number of detection:',detector_output['num_detections'][0].numpy())
  print("Detection class label:",detector_output['detection_classes'][0].numpy())
  #print('Detection boxes coordinates:',detector_output['detection_boxes'])
  print('Detection scores:',detector_output['detection_scores'][0].numpy())

#===================================================================
# function: load model 
def load_model(model_path):
    with warnings.catch_warnings():
        # Filter all DeprecationWarnings
        warnings.filterwarnings('ignore')
        try:
            object_detection_model = tf.saved_model.load(model_path)
            return object_detection_model
        except:
            print('Model path is not Found!')
            return None
#========================================================================


#==============================================================
# function: to get time range to perfrom blur
def time_range():
    start_time = float(input('Enter start time(minutes): '))
    end_time = float(input('Enter end time(minutes):' ))
    start_time,end_time = start_time*60,end_time*60 # change to second(s) format
    return start_time,end_time
#===================================================================

#====================================================================
# function: to check if time range is valid or not
def is_valid_time_range(start_time,end_time,video_duration):
    return (0 <= start_time < end_time <= video_duration)
#=================================================================

#=====================================================================
# function: to process video
def process_video(video_path,start_time,end_time,select_classes):
    print('processing video')
    try:
        cap = cv2.VideoCapture(video_path)
        print('Video loaded successfully!')
    except:
        print("Failed! to load video")
    # get video property like frame_width,frame_heigh,frame_per_second(fps),codecc
    frame_width = int(cap.get(3)) # width of the fames in the video
    frame_height = int(cap.get(4)) # height of the frame in the video
    fps = int(cap.get(5)) # frame per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration =  total_frames/ fps
    codecc = cv2.VideoWriter_fourcc(*'XVID') # codecc for output video (XVID default codecc)
        
    # VideoWriter object to save blured video
    op_video_path = 'car_blured_op_video.mp4'
    out = cv2.VideoWriter(op_video_path,codecc,fps,(frame_width,frame_height))
    
    # load the object detection model
    object_detection_model = load_model('efficientdet_v2')
    # Loop through each video frame,blured the image and write in output video
    if (is_valid_time_range(start_time,end_time,video_duration)):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        for i in range(start_frame,end_frame):
            ret, frame = cap.read() # Read frame from the input image
          #print('Original frame shape:',frame.shape)
          #print('fame shape:',frame.shape)
          # Expand frame dimension
            if ret:
                frame = tf.expand_dims(frame,axis = 0)
            else:
                break # break if we have reached the end of the video
              # Blur the image
            detector_output = object_detection_model(frame)
            boxes,classes,scores = filter_detection(detector_output,select_classes)
    
          #print('New frame shape:',frame[0].shape)
            blured_img = blur_image(frame[0].numpy(),boxes)
          #print('blured_img_shape',blured_img.shape)
          # write blured frame to output image
            out.write(blured_img)
          # Display original and blured frame
          #show_img(frame[0].numpy(),'Original_frame')
          #show_img(blured_img,'Blured_frame')
        # release video capture video write object
        cap.release()
        out.release()
    
        # close all opencv windows
        cv2.destroyAllWindows()
    else:
        
        while True:
          ret, frame = cap.read() # Read frame from the input image
          #print('Original frame shape:',frame.shape)
          #print('fame shape:',frame.shape)
          # Expand frame dimension
          if ret:
            frame = tf.expand_dims(frame,axis = 0)
          else:
            break # break if we have reached the end of the video
          # Blur the image
          detector_output = object_detection_model(frame)
          boxes,classes,scores = filter_detection(detector_output,select_classes)
    
          #print('New frame shape:',frame[0].shape)
          blured_img = blur_image(frame[0].numpy(),boxes)
          #print('blured_img_shape',blured_img.shape)
          # write blured frame to output image
          out.write(blured_img)
          # Display original and blured frame
          #show_img(frame[0].numpy(),'Original_frame')
          #show_img(blured_img,'Blured_frame')
    
    
          #break the loop if key 'q' is pressed
          #if cv2.waitKey(1) & 0xFF == ord('q'):
           # break
    
    
        # realease video capture and writer object
        cap.release()
        out.release()
    
        # close all opencv windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('runing in actual script')
    video_path = 'car_blur_test.mp4'
    start_time,end_time = time_range()
    select_classes =  [3]
    process_video(video_path, start_time, end_time, select_classes)
    

