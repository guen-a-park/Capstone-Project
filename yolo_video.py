from fileinput import filename
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

config_dict = {}

#'C:/Users/kate1/capstone/keras-yolo3/model_data/trained_weights_stage_1.h5'

yolo2 = YOLO(model_path='C:/Users/kate1/capstone/keras-yolo3/model_data/helmet.h5',
            anchors_path='C:/Users/kate1/capstone/keras-yolo3/model_data/yolo_anchors.txt',
            classes_path='C:/Users/kate1/capstone/keras-yolo3/model_data/helmet_classes.txt',
           score=0.3)


FLAGS = None

img = Image.open(os.path.join('C:/Users/kate1/capstone/helmet_test.PNG'))
img_copy = img.copy()
detected_img,label,left,top,right,bottom = yolo2.detect_image(img)
#print(label,left,top,right,bottom) 
result=np.asarray(detected_img)

area =(left,top,right,bottom)
bdbox = img_copy.crop(area) #image[ystart:ystop, xstart:xstop] left, upper, right, lower
if label == "NonHelmet": 
    bdbox.show()
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()

else:
    print("헬멧을 착용했습니다.")

#######################################################

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

pytesseract.pytesseract.tesseract_cmd = R'C:/Program Files/Tesseract-OCR/tesseract.exe'

min_confidence = 0.5
file_name = 'C:/Users/kate1/capstone/keras-yolo3/testcar5.jpg' #testcar5

east_decorator = 'C:/Users/kate1/capstone/keras-yolo3/frozen_east_text_detection.pb'

frame_size = 320
padding = 0.05

def textROI(image_cr):
    # load the input image and grab the image dimensions
    
    image_cr = cv2.cvtColor(np.array(image_cr), cv2.COLOR_RGB2BGR)

    orig = image_cr.copy()
    (origH, origW) = image_cr.shape[:2]


    rW = origW / float(frame_size)
    rH = origH / float(frame_size)
    newW = int(origW / rH)
    center = int(newW / 2)
    start = center - int(frame_size / 2)

    # resize the image and grab the new image dimensions
    image_cr = cv2.resize(image_cr, (newW, frame_size))
    scale_image = image_cr[0:frame_size, start:start+frame_size]
    (H, W) = scale_image.shape[:2]

    #중간점검
    cv2.imshow("orig", orig)
    cv2.imshow("resize", image_cr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("scale_image", scale_image)

    # define the two output layer names for the EAST detector model
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_decorator)

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image_cr, 1.0, (frame_size, frame_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities)
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):

                if scoresData[x] < min_confidence:
                        continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

    # apply non-maxima suppression
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            return ([startX, startY, endX, endY], orig[startY:endY, startX:endX])

def textRead(image):
    # apply Tesseract v4 to OCR
    #config = ("-l eng --oem 1 --psm 7") #7 to 4  kor+eng
    #config = ("digits --oem 3 --psm 3")
    config = ('--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    text = pytesseract.image_to_string(image, config=config)

    #display the text OCR'd by Tesseract
    print("OCR TEXT : {}\n".format(text))

    # strip out non-ASCII text
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    #print("Alpha numeric TEXT : {}\n".format(text))
    return text

# Loading image
([startX,startY,endX,endY], text_image) = textROI(bdbox)
cv2.imshow('plate_img',text_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
text = textRead(text_image)
image_cr = cv2.cvtColor(np.array(bdbox), cv2.COLOR_RGB2BGR)

# show the output image
cv2.imshow("OCR Text Recognition : "+text, image_cr)
cv2.waitKey(0)
cv2.destroyAllWindows()










#####################################################################

#for video
def detect_img(yolo2):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo2.detect_image(image)
            r_image.show()
    yolo.close_session()




# C:/Users/kate1/capstone/keras-yolo3/testcar_video.mp4
# python yolo_video.py --input C:/Users/kate1/capstone/keras-yolo3/test_video.mp4



#yolo2.detect_video
# if __name__ == '__main__':
#     # class YOLO defines the default value, so suppress any default here
#     parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
#     '''
#     Command line options
#     '''
#     parser.add_argument(
#         '--model', type=str,
#         help='path to model weight file, default ' + YOLO.get_defaults("model_path")
#     )

#     parser.add_argument(
#         '--anchors', type=str,
#         help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
#     )

#     parser.add_argument(
#         '--classes', type=str,
#         help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
#     )

#     parser.add_argument(
#         '--gpu_num', type=int,
#         help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
#     )

#     parser.add_argument(
#         '--image', default=False, action="store_true",
#         help='Image detection mode, will ignore all positional arguments'
#     )
#     '''
#     Command line positional arguments -- for video detection mode
#     '''
#     parser.add_argument(
#         "--input", nargs='?', type=str,required=False,default='./path2your_video',
#         help = "Video input path"
#     )

#     parser.add_argument(
#         "--output", nargs='?', type=str, default="",
#         help = "[Optional] Video output path"
#     )

#     FLAGS = parser.parse_args()

#     if FLAGS.image:
#         """
#         Image detection mode, disregard any remaining command line arguments
#         """
#         print("Image detection mode")
#         if "input" in FLAGS:
#             print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
#         # detect_img(yolo2(**vars(FLAGS)))
#         detect_img(YOLO(**vars(FLAGS)))
#     elif "input" in FLAGS:
#         detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output) #이게 예전 그대로 아닌가
#         # detect_video(yolo2(**vars(FLAGS)), FLAGS.input, FLAGS.output) #이게 예전 그대로 아닌가
            
#     else:
#         print("Must specify at least video_input_path.  See usage with --help.")
