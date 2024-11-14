from scipy.cluster.vq import kmeans
import pandas as pd
from scipy.cluster.vq import whiten
from matplotlib import pyplot as plt
from matplotlib import image as img
import numpy as np
import cv2 as cv

def CalculateHistogramm(imageL):
    colums = imageL.shape[0]
    rows = imageL.shape[1]
    hist = np.zeros(180)
    for m in range(0, colums - 1, 1):
        for n in range(0, rows - 1, 1):
            hist[imageL[m, n]] = hist[imageL[m, n]] + 1
    return hist

def ContrustUp(imageL):
    hist = CalculateHistogramm(imageL)
    histEqulizer = np.zeros(180)
    colums = imageL.shape[0]
    rows = imageL.shape[1]
    histEqulizer[0] = hist[0]
    for k in range(1, 179, 1):
        histEqulizer[k] = histEqulizer[k - 1] + hist[k]
    plt.plot(range(180), histEqulizer)
    plt.show()
    for k in range(1, 179, 1):
        if histEqulizer[k] > 0:
            histEqulizerMin = histEqulizer[k]
            break
    L = np.zeros(180)
    for l in range(1, 179, 1):
        L[l] = round(((histEqulizer[l] - histEqulizerMin)/((colums * rows) - 1)) * 179)
    imageNew = imageL
    for m in range(colums - 1):
        for n in range(rows - 1):
            imageNew[m,n] = L[imageNew[m,n]]
    return imageNew

def ImageFonFilter(imageL):
    colums = imageL.shape[0]
    rows = imageL.shape[1]
    thDefined, imageBinary = cv.threshold(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 0, 255,
                                          cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('', imageBinary)
    cv.waitKey()
    imageNew = np.empty((image.shape[0], image.shape[1], 3))
    imageNew = cv.bitwise_and(imageL, imageL, imageBinary, imageBinary)
    return imageNew

if __name__ == "__main__":

    image = cv.imread('C:/Users/ACER1/Desktop/Computer Vision/test1.jpg', cv.IMREAD_COLOR)
    cv.imshow('', image)
    cv.waitKey()

    image = cv.bilateralFilter(image, 5, 75, 75)
    cv.imshow('', image)
    cv.waitKey()

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv.filter2D(image, -1, kernel)
    cv.imshow('', image)
    cv.waitKey()

    image = ImageFonFilter(image)
    cv.imshow('', image)
    cv.waitKey()

    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('', image[:, :, 0])
    cv.waitKey()

    histogramm = CalculateHistogramm(image[:, :, 0])
    histogramm[0] = 0
    plt.plot(range(180), histogramm)
    plt.show()
    pass

