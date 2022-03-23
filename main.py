import cv2 as cv
import DescriptorMatch
import FonctionsUtilitaires as util
import matplotlib.pyplot as plt

basedir = 'DataKITTI'
date = '2011_09_26'
drive = '0019'
#distance entre caméras: 0.54m
sensorWidth = 6.4*25.4
sensorHeight = 4.8*25.4

#img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)  # queryImage
#img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage
img1 = cv.imread('left.png', cv.IMREAD_GRAYSCALE)           # queryImage
img2 = cv.imread('right.png', cv.IMREAD_GRAYSCALE)          # trainImage
print(img1.shape)
matches, kp1, kp2 = DescriptorMatch.DescriptorMatch(img1, img2)
matches_bin = DescriptorMatch.MatchesBinning(matches, kp1, 16, img1.shape, 1000)

test = kp1[0]
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches_bin, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()


#On doit calculer la distance Z des keypoints avec les images stereo pour chaque step k,
# et calculer deltaZ entre k-1 et k pour determiner la distance parcourue a chaque step k

