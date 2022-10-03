import cv2 as cv
import numpy as np

# Setup
yolo = cv.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
classes = []
with open("./coco.names", 'r') as f:
    classes = f.read().splitlines()

# Image
img = cv.imread("./Images/othercat.jpg")

# Format the image, first convert integer value to range [0,1], and resize the iamge
blob = cv.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
print(blob.shape)
# To print image
i = blob[0].reshape(320, 320, 3)
cv.imshow('Cat', i)


# set blob as input image
yolo.setInput(blob)
output_layers_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_name)

# Create rrays to store the objects detected and their probabilities
boxes = []
confidences = []
class_ids = []
height = img.shape[0]
width = img.shape[1]

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# To check for output
# print(len(boxes))
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size = (len(boxes),3))
for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i],2))
    color = colors[i]

    cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv.putText(img, label +" "+confi, (x,y+20), font, 2, (255,255,255), 2)

# Output Window
cv.namedWindow("Output", cv.WINDOW_NORMAL)
cv.resizeWindow("Output", width, height)
cv.imshow('Output', img)
cv.waitKey(0)