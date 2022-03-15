import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import DescriptorMatch

#distance entre caméras: 0.54m

img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage
# img1 = cv.imread('left.png',cv.IMREAD_GRAYSCALE)           # queryImage
# img2 = cv.imread('right.png',cv.IMREAD_GRAYSCALE)          # trainImage

matches, kp1, kp2 = DescriptorMatch.DescriptorMatch(img1, img2)
test = kp1[0]
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3), plt.show()
