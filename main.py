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
    for m in range(colums):
        for n in range(rows):
            hist[imageL[m, n]] = hist[imageL[m, n]] + 1
    return hist

def ContrustUp(imageL):
    hist = CalculateHistogramm(imageL)
    histEqulizer = np.zeros(180)
    colums = imageL.shape[0]
    rows = imageL.shape[1]
    histEqulizer[0] = hist[0]
    for k in range(1, 180, 1):
        histEqulizer[k] = histEqulizer[k - 1] + hist[k]
    for k in range(1, 180, 1):
        if histEqulizer[k] > 0:
            histEqulizerMin = histEqulizer[k]
            break
    L = np.zeros(180)
    for l in range(1, 180, 1):
        L[l] = round(((histEqulizer[l] - histEqulizerMin)/((colums * rows) - 1)) * 180)
    imageNew = imageL
    for m in range(colums - 1):
        for n in range(rows - 1):
            imageNew[m,n] = L[imageNew[m,n]]
    return imageNew

if __name__ == "__main__":
    image = cv.imread('C:/Users/ACER1/Desktop/Computer Vision/test1.jpg', cv.IMREAD_COLOR)
    image = cv.bilateralFilter(image, 5, 75, 75)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv.filter2D(image, -1, kernel)
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    cv.imshow('', image[:, :, 0])
    cv.waitKey()
    imageNew = ContrustUp(image[:, :, 0])
    cv.imshow('', image[:, :, 0])
    cv.waitKey()
    plt.plot(range(256), CalculateHistogramm(image[:, :, 0]))
    plt.show()
    pass

