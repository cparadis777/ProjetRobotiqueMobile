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
sequence = '00'
data = pk.odometry(basedir, sequence)

fx = data.calib.K_cam0[0, 0]
fy = data.calib.K_cam0[1, 1]
b = data.calib.b_gray
sensorWidth = 6.4*25.4
sensorHeight = 4.8*25.4

orb = cv.ORB_create()

pose = [np.array([0, 0, 0])]
poseRef = [np.array([0, 0, 0])]
transfo_cumulee = [np.identity(4)]
list_transfo_ref = [data.poses[0]]



for i in range(1, len(data.poses)):
#for i in range(1, 200):
    transfo = disp.transformationStep(i, i - 1, fx, fy, b, data, orb, 'icp', False)
    #transfo_cumulee.append(np.matmul(transfo_cumulee[i-1], transfo))
    #poseStep = transfo_cumulee[i][0:3, 3]

    poseStep = transfo[0:3, 3]
    poseStep = np.add(pose[i-1], poseStep)
    list_transfo_ref.append(data.poses[i])

    #print(transfo[0:3, 3])
    # print('Transfo - transfo_Ref')
    # print(transfo)
    # print(data.poses[i])
    #print(np.add(transfo, -1*data.poses[i]))

    poseRefStep = data.poses[i][0:3, 3]
    # print("calculee", poseStep)
    # print('ref', poseRefStep)
    # print('diff', np.add(poseRefStep, -1*poseStep))
    # print('stepSizeRef', np.add(poseRefStep, -1*poseRef[i-1]))
    # print('stepSizecalc', np.add(poseStep, -1*pose[i-1]))

    #print(data.poses[i])
    #print(transfo)

    pose.append(poseStep)
    poseRef.append(poseRefStep)
    print(" step {}/{}".format(i, len(data.poses)-1))
    #break


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

for i in poseRef:
    xref.append(i[0])
    yref.append(i[1])
    zref.append(i[2])


fig = plt.figure()

ax = fig.add_subplot(211)
plt.plot(np.dot(1, zref), np.dot(-1, xref))


ax = fig.add_subplot(212)
plt.plot(z, np.dot(-1, x))

plt.tight_layout()
plt.show()
print('end')

# img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3), plt.show()


# On doit calculer la distance Z des keypoints avec les images stereo pour chaque step k,
# et calculer deltaZ entre k-1 et k pour determiner la distance parcourue a chaque step k
