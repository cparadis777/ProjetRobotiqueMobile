import FonctionsUtilitaires as util
import numpy as np
import cv2 as cv
import pcl

# TODO: clean-up, c'est assez spaghetti
def transformationStep(step, stepPrecedent, f, b, data, orb):
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
        z = util.distanceFromStereoPoints(kp1G[trainIdx].pt, kp1D[queryIdx].pt, f, b)
        coord = util.get3Dcoord(kp1G[trainIdx].pt, z, f)
        coords1[kp1G[trainIdx]] = coord

    for i in matches2GD:
        trainIdx = i.trainIdx
        queryIdx = i.queryIdx
        z = util.distanceFromStereoPoints(kp2G[trainIdx].pt, kp2D[queryIdx].pt, f, b)
        coord = util.get3Dcoord(kp2G[trainIdx].pt, z, f)
        coords2[kp2G[trainIdx]] = coord

    for i in matchesTemporel[:nbpoints]:
        if kp1G[i.trainIdx] in coords1 and kp2G[i.queryIdx] in coords2:
            points1.append(coords1[kp1G[i.trainIdx]])
            points2.append(coords2[kp2G[i.queryIdx]])

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    points1 = np.float32(points1[:, np.newaxis, :])
    points2 = np.float32(points2[:, np.newaxis, :])

    retval, transfo, inliers = cv.estimateAffine3D(points1, points2, ransacThreshold=3)
    return transfo
