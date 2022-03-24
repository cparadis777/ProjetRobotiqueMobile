import FonctionsUtilitaires as util

import cv2 as cv

# TODO: clean-up, c'est assez spaghetti
def transformationStep(step, stepPrecedent, f, b, data, orb):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    img1G = util.KITTI2OpenCV(data.get_cam0(stepPrecedent))  # queryImage
    img1D = util.KITTI2OpenCV(data.get_cam1(stepPrecedent))  # trainImage
    img2G = util.KITTI2OpenCV(data.get_cam0(step))  # queryImage
    img2D = util.KITTI2OpenCV(data.get_cam1(step))  # trainImage

    kp1G, desc1G = orb.detectAndCompute(img1G, None)
    kp1D, desc1D = orb.detectAndCompute(img1D, None)
    kp2G, desc2G = orb.detectAndCompute(img2G, None)
    kp2D, desc2D = orb.detectAndCompute(img2D, None)

    matches1GD = sorted(bf.match(desc1G, desc1D), key=lambda x: x.distance)
    matches2GD = sorted(bf.match(desc2G, desc2D), key=lambda x: x.distance)
    matchesTemporel = sorted(bf.match(desc1G, desc2G), key=lambda x: x.distance)

    nbpoints = min(len(matches1GD), len(matches2GD), len(matchesTemporel))
    points1 = []
    points2 = []
    for i in range(nbpoints):

    # cv.findHomography()
    return 1
