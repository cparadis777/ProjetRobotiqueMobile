# Defini des fonctions permettant de convertir
# les paramètres des caméras de [m] a [pixels]

import math


# Converti Fx, Fy, X0 ou Y0 en fx, fy, x0 ou y0
def convertReal2Pixel(paramReel, dimReelle, dimImg):
    return paramReel * (dimReelle / dimImg)


def distanceFromStereoPoints(coordG, coordD, f, b):
    d = math.hypot(coordD[0] - coordG[0], coordD[1] - coordG[1])
    return b * f / d
