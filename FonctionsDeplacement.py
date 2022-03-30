import math

# noinspection PyUnresolvedReferences
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


import FonctionsUtilitaires as util
import Rigid3Dtransform as rigid


def generatePointClouds(step, stepPrecedent, fx, fy, b, data, orb):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    img1G = util.KITTI2OpenCV(data.get_cam0(stepPrecedent))  # queryImage step k-1
    img1D = util.KITTI2OpenCV(data.get_cam1(stepPrecedent))  # trainImage step k-1
    img2G = util.KITTI2OpenCV(data.get_cam0(step))  # queryImage step k
    img2D = util.KITTI2OpenCV(data.get_cam1(step))  # trainImage step k

    kp1G, desc1G = orb.detectAndCompute(img1G, None)
    kp1D, desc1D = orb.detectAndCompute(img1D, None)
    kp2G, desc2G = orb.detectAndCompute(img2G, None)
    kp2D, desc2D = orb.detectAndCompute(img2D, None)

    matches1GD = sorted(bf.match(desc1G, desc1D), key=lambda x: x.distance)
    matches2GD = sorted(bf.match(desc2G, desc2D), key=lambda x: x.distance)
    matchesTemporel = sorted(bf.match(desc1G, desc2G), key=lambda x: x.distance)

    nbpoints = min(len(matches1GD), len(matches2GD), len(matchesTemporel))

    coords1 = {}
    coords2 = {}
    points1 = []
    points2 = []

    for i in matches1GD:
        trainIdx = i.trainIdx
        queryIdx = i.queryIdx
        z = util.distanceFromStereoPoints(kp1G[trainIdx].pt, kp1D[queryIdx].pt, fx, b)
        coord = util.get3Dcoord(kp1G[trainIdx].pt, z, fx, fy)
        coords1[kp1G[trainIdx]] = coord

    for i in matches2GD:
        trainIdx = i.trainIdx
        queryIdx = i.queryIdx
        z = util.distanceFromStereoPoints(kp2G[trainIdx].pt, kp2D[queryIdx].pt, fx, b)
        coord = util.get3Dcoord(kp2G[trainIdx].pt, z, fx, fy)
        coords2[kp2G[trainIdx]] = coord

    for i in matchesTemporel[:nbpoints]:
        if kp1G[i.trainIdx] in coords1 and kp2G[i.queryIdx] in coords2:

            points1.append(coords1[kp1G[i.trainIdx]])
            points2.append(coords2[kp2G[i.queryIdx]])

            #print('k: {}, k+1: {}'.format(coords1[kp1G[i.trainIdx]], coords2[kp2G[i.queryIdx]]))

    mask = []
    for i in range(len(points1)):
        distance = util.euclidian_distance(points1[i], points2[i])
        if distance <= 5:
            mask.append(True)
        else:
            mask.append(False)

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)

    points1 = points1[mask]
    points2 = points2[mask]
    return points1, points2, coords1, coords2


# TODO: clean-up, c'est assez spaghetti
def transformationStep(step, stepPrecedent, fx, fy, b, data, orb, type, draw):


    points1, points2, coords1, coords2 = generatePointClouds(step, stepPrecedent, fx, fy, b, data, orb)

    #print(util.euclidian_distance(points1[0], points2[0]))
    top50percent = int(math.ceil(len(points1)))
    top3 = 40
    #points1 = points1[0:top3, :]
    #points2 = points2[0:top3, :]
    points3 = np.transpose(points1)
    points4 = np.transpose(points2)
    points1 = np.float32(points1[:, np.newaxis, :])
    points2 = np.float32(points2[:, np.newaxis, :])

    if type == 'affine':
        retval, transfo, inliers = cv.estimateAffine3D(points1, points2, ransacThreshold=3, confidence=0.98)
        transfo = np.vstack([transfo, np.transpose(np.array([0, 0, 0, 1])[:, None])])

    elif type == 'rigid':
        transfo = rigid.rigid_transform_3D(points3, points4)


    else:
        raise Exception('Type de transformation invalide')



    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        k = 0
        for i in coords1:
            if k > 10:
                pass
            else:
                coord_homo = np.array([coords1[i][0], coords1[i][1], coords1[i][2], 1])
                point_transfo = np.matmul(np.linalg.inv( transfo), coord_homo)
                #ax.scatter(coords1[i][0], coords1[i][1], coords1[i][2], color='red')
                ax.scatter(point_transfo[0], point_transfo[1], point_transfo[2], color='green')
            k = k + 1
        k = 0
        for i in coords2:
            if k > 10:
                pass
            else:
                coord_homo = np.array([coords2[i][0], coords2[i][1], coords2[i][2], 1])
                point_transfo = np.matmul(transfo, coord_homo)
                ax.scatter(coords2[i][0], coords2[i][1], coords2[i][2], color='blue')
                #ax.scatter(point_transfo[0], point_transfo[1], point_transfo[2], color='orange')
            k = k + 1
        plt.show()

    return transfo
