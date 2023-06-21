import cv2
import numpy as np
from PIL import Image
import sys
import os

if len(sys.argv) < 2:
    print("Please specify the image path")
    print("Usage: python clean_png.py <image_path> <output_path>")
    exit()

img_path = sys.argv[1]

img = Image.open(img_path).convert('RGBA')
# set all alpha values that are lower than 100 to 0, keep all the others
alpha = img.getchannel("A")

alpha = alpha.point(lambda i: 0 if i<100 else i)

# merge the new alpha channel with the old image
img.putalpha(alpha)

#if output path is not specified, save to output.png
#while output.png already exists, save to output1.png, output2.png, etc.

if len(sys.argv) < 3:
    output_path = 'output.png'
    if os.path.exists(output_path):
        i = 1
        while os.path.exists(output_path):
            output_path = 'output' + str(i) + '.png'
            i += 1
else:
    output_path = sys.argv[2]

img.save(output_path)