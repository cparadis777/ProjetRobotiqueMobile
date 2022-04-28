# Defini des fonctions permettant de convertir
# les paramètres des caméras de [m] a [pixels]
import cv2 as cv
import math
import numpy as np
#from scipy.spatial import distance

# Convert Fx, Fy, X0 ou Y0 en fx, fy, x0 ou y0
def convertReal2Pixel(paramReel, dimReelle, dimImg):
    return paramReel * (dimReelle / dimImg)


def distanceFromStereoPoints(coordG, coordD, f, b):
    d = math.hypot(coordD[0] - coordG[0], coordD[1] - coordG[1])
    return b * f / d


def KITTI2OpenCV(img):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def get3Dcoord(coordIm, Z, fx, fy):
    x = coordIm[0] * Z / fx
    y = coordIm[1] * Z / fy
    return (x, y, Z)

def euclidian_distance(points1, points2):
    distance = math.sqrt((points2[0]-points1[0])**2+(points2[1]-points1[1])**2+(points2[2]-points1[2])**2)
    return distance

