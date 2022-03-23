import math
import cv2 as cv


def DescriptorMatch(img1, img2):
    # Création d'un objet descripteur ORB
    orb = cv.ORB_create()
    # Détection des keypoints dans chacune des images et calcul des descripteurs
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Création d'un objet BFMatcher avec un appariement basé sur la norme de Hamming et avec l'option
    # crossCheck (s'assure d'un match optimal bidirectionnel)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Appariement des descripteurs.
    matches = bf.match(des1, des2)
    # Classement croissant des paires par leur distance.
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp1, kp2


def MatchesBinning(matches, kp1, nbBins, dimensionImage, nbMatchBin):
    if dimensionImage[0] % math.sqrt(nbBins) != 0 or dimensionImage[1] % math.sqrt(nbBins) != 0:
        raise Exception("Impossible de séparer l'image en {} bins".format(nbBins))
    else:
        racineNbBins = int(math.sqrt(nbBins))
        binned_matches = []
        sizeBinX = dimensionImage[0] / math.sqrt(nbBins)
        sizeBinY = dimensionImage[1] / math.sqrt(nbBins)
        frontieresX = [i * sizeBinX for i in range(racineNbBins + 1)]
        frontieresY = [i * sizeBinY for i in range(racineNbBins + 1)]
        for i in range(racineNbBins):  # iteration en x
            binsHor = []
            for j in range(racineNbBins):  # iteration en y
                kpInBin = []
                for k in range(len(matches)):
                    if (frontieresX[i] <= kp1[matches[k].trainIdx].pt[0] <= frontieresX[i + 1] or
                            frontieresY[i] <= kp1[matches[k].trainIdx].pt[1] <= frontieresY[i + 1]):
                        kpInBin.append(matches[k])
                binsHor.append(kpInBin)
            binned_matches.append(binsHor)
        binned_matches = sortBinnedMatches(binned_matches, nbMatchBin)
    return binned_matches


def sortBinnedMatches(binned_matches, nbMatchBin):
    sorted_matches = []
    for i in binned_matches:
        for j in i:
            temp = sorted(j, key=lambda x: x.distance)
            for k in temp[1:nbMatchBin]:
                sorted_matches.append(k)
    sorted_matches = sorted(sorted_matches, key=lambda x: x.distance)
    return sorted_matches
