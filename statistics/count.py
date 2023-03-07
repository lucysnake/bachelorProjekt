import numpy as np
import cv2 as cv
from PIL import Image
import cv2
import numpy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)
mini = 1000
maxx = 0
summe = 0
count = 0
import cv2
import numpy as np
import matplotlib.pyplot as plt
asdf = []
for label in os.listdir("Images/train/labels/"):
    # Reading an image in default mode:
    inputImage = cv2.imread('Images/train/labels/' + label)
    # Deep copy for results:
    inputImageCopy = inputImage.copy()
    # Convert RGB to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    # Find the contours on the binary image:
    contours, hierarchy = cv2.findContours(grayscaleImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Store bounding rectangles and object id here:
    objectData = []
    # ObjectCounter:
    objectCounter = 0
    # Look for the outer bounding boxes (no children):
    for _, c in enumerate(contours):
        # Get the contour's bounding rectangle:
        boundRect = cv2.boundingRect(c)

        # Store in list:
        objectData.append((objectCounter, boundRect))

        # Get the dimensions of the bounding rect:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Draw bounding rect:
        color = (0, 0, 255)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                    (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

        # Draw object counter:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontThickness = 2
        color = (0, 255, 0)
        cv2.putText(inputImageCopy, str(objectCounter), (int(rectX), int(rectY)), 
                    font, fontScale, color, fontThickness)

        # Increment object counter
        objectCounter += 1

        #cv2.imshow("Rectangles", inputImageCopy)
        #cv2.waitKey(0)
    print(str(objectCounter) + ' ' + label)
    asdf.append(objectCounter)


print(sum(asdf))

for i in range(0,23):
    print(asdf.count(i))

"""
for label in os.listdir("data/validation/labels/"):
    tsr = torchvision.io.read_image('data/validation/labels/' + label)
    tsr = asdf(tsr)
    count += 1
    a = tsr.unique(return_counts=True)
    print(a)
"""
    

"""
for label in os.listdir("data/train/labels/"):
    tsr = torchvision.io.read_image('data/train/labels/' + label)
    tsr = asdf(tsr)
    a = tsr.unique(return_counts=True)
    count += 1
    if a[1][0].item() == 262144: #wenn alle pixel 0 sind
        continue
    area = a[1][1].item()
    summe += area
    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area
"""
#https://stackoverflow.com/questions/59525065/detecting-object-location-in-image-with-python-opencv object location