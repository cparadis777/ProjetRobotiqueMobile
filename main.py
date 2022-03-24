import cv2 as cv
import numpy
import numpy as np

import DescriptorMatch
import pykitti as pk
import FonctionsUtilitaires as util
import matplotlib.pyplot as plt
import FonctionsDeplacement as disp

basedir = 'DataKITTI/dataset'
sequence = '00'
data = pk.odometry(basedir, sequence)

fx = data.calib.K_cam0[0, 0]
fy = data.calib.K_cam0[1, 1]
b = 0.54*1000
sensorWidth = 6.4*25.4
sensorHeight = 4.8*25.4

orb = cv.ORB_create()
test = disp.transformationStep(0, 1, fx, b, data, orb)
pose = [np.array([0, 0, 0])]

for i in range(1, 100):
    disp.transformationStep(i, i-1, fx, b, data, orb)
    displacement = numpy.matmul(test, np.array([0, 0, 0, 1]))
    pose.append(np.add(pose[i-1], displacement))


print('end')


#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3), plt.show()


#On doit calculer la distance Z des keypoints avec les images stereo pour chaque step k,
# et calculer deltaZ entre k-1 et k pour determiner la distance parcourue a chaque step k

