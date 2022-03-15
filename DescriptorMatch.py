import math

import cv2 as cv


def DescriptorMatch(img1, img2):
    # Création d'un objet descripteur ORB
    orb = cv.ORB_create()
    # Détection des keypoints dans chacune des images et calcul des descripteurs
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Création d'un objet BFMatcher avec un appariement basé sur la norme de Hamming et avec l'option
    # crossCheck (S'assure d'un match optimal bidirectionnel)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Appariement des descripteurs.
    matches = bf.match(des1, des2)
    # Classement croissant des paires par leur distance.
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp1, kp2


def MatchesBinning(matches, kp1, kp2, nbBins, dimensionImage):
    binned_matches = []
    if math.sqrt(nbBins) != 0:
        raise Exception("Impossible de séparer l'image en {} bins".format(nbBins))
    else:
        sizeBinX = dimensionImage[0]/math.sqrt(nbBins)
        sizeBinY =  dimensionImage[1]/math.sqrt(nbBins)
        frontieresX = []
        frontieresY = []

    return binned_matches
