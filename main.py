import cv2 as cv
import DescriptorMatch
import pykitti as pk
import FonctionsUtilitaires as util
import matplotlib.pyplot as plt

basedir = 'DataKITTI/dataset'
sequence = '00'
data = pk.odometry(basedir, sequence)


#distance entre caméras: 0.54m
sensorWidth = 6.4*25.4
sensorHeight = 4.8*25.4
plt.imshow(data.get_cam0(1))
plt.show()
plt.imshow(data.get_cam0(2))
plt.show()
#img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)  # queryImage
#img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage
img1 = cv.imread('left.png', cv.IMREAD_GRAYSCALE)           # queryImage
img2 = cv.imread('right.png', cv.IMREAD_GRAYSCALE)          # trainImage
print(img1.shape)
matches, kp1, kp2 = DescriptorMatch.DescriptorMatch(img1, img2)
matches_bin = DescriptorMatch.MatchesBinning(matches, kp1, 16, img1.shape, 1000)

test = kp1[0]
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches_bin, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3), plt.show()


#On doit calculer la distance Z des keypoints avec les images stereo pour chaque step k,
# et calculer deltaZ entre k-1 et k pour determiner la distance parcourue a chaque step k

