import cv2
import numpy as np
from PIL import Image

img = Image.open('IMG_9013.png').convert('RGBA')
# set all alpha values that are lower than 100 to 0, keep all the others
alpha = img.getchannel("A")

alpha = alpha.point(lambda i: 0 if i<100 else i)

# merge the new alpha channel with the old image
img.putalpha(alpha)


img.save('output2.png')