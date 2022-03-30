import cv2 as cv
import matplotlib.pyplot
import numpy
import numpy as np

import DescriptorMatch
import pykitti as pk
import FonctionsUtilitaires as util
import matplotlib.pyplot as plt
import FonctionsDeplacement as disp

basedir = 'DataKITTI/dataset'
sequence = '06'
data = pk.odometry(basedir, sequence)

fx = data.calib.K_cam0[0, 0]
fy = data.calib.K_cam0[1, 1]
b = data.calib.b_gray
sensorWidth = 6.4*25.4
sensorHeight = 4.8*25.4

orb = cv.ORB_create()
test = disp.transformationStep(0, 1, fx, b, data, orb, False)


pose = [np.array([0, 0, 0])]
pose2 = [np.array([0, 0, 0])]
# print("test \n", test)
# print('calib \n', data.calib.K_cam0)
# test2 = np.matmul(data.calib.K_cam0, test)
# print(test2)
# test3 = np.matmul(test2, np.array([0,0,0,1]))
#
# print(test3)
taille_sample = len(data.cam0_files)
for i in range(1, 500):
    try:
        transfo = disp.transformationStep(i, i-1, fx, b, data, orb, False)
        #transfo = np.vstack([transfo, np.array([0, 0, 0, 1])])
        transfoRef = data.poses[i]
        poseRef = transfoRef[0:3, 3]
        #print(transfo[0:3, 3])
        posestep = transfo[0:3, 3]
        #print(posestep)
        #posestep = np.matmul(np.linalg.inv(data.calib.K_cam0), posestep)
        #posestep = np.matmul(data.calib.K_cam0, posestep)
        pose2.append(transfoRef[0:3, 3])
        pose.append(np.add(pose[i-1], posestep))
        #pose.append(posestep)
        print("{} / {} : ref {}, cal {}".format(i, taille_sample-1, poseRef, posestep))
    except:
        pass

fig = plt.figure()
x = []
y = []
z = []
xref = []
yref = []
zref = []
for i in pose:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
for i in pose2:
    xref.append(i[0])
    yref.append(i[1])
    zref.append(i[2])

plt.subplot(321)
plt.plot(z, y)
plt.subplot(323)
plt.plot(x, z)
plt.subplot(325)
plt.plot(z, x)

plt.subplot(322)
plt.plot(zref, yref)
plt.subplot(324)
plt.plot(xref, zref)
plt.subplot(326)
plt.plot(zref, np.dot(-1, xref))

plt.show()

print('end')


#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3), plt.show()


#On doit calculer la distance Z des keypoints avec les images stereo pour chaque step k,
# et calculer deltaZ entre k-1 et k pour determiner la distance parcourue a chaque step k

