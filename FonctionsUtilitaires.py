# Defini des fonctions permettant de convertir
# les paramètres des caméras de [m] a [pixels]
import cv2 as cv
import math
import numpy as np
#from scipy.spatial import distance




def distanceFromStereoPoints(coordG, coordD, f, b):
    d = coordG[1] - coordD[1]
    return b * f / d


def KITTI2OpenCV(img):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def get3Dcoord(coordIm, Z, fx, fy):
    X = coordIm[0] * Z / fx
    Y = coordIm[1] * Z / fy
    return X, Y, Z

def euclidian_distance(points1, points2):
    distance = math.sqrt((points2[0]-points1[0])**2+(points2[1]-points1[1])**2+(points2[2]-points1[2])**2)
    return distance

