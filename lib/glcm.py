import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops


class GLCM(object):
    def __init__(self, image, ):
        self.img = image
        self.dists = [5]
        self.agls = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.lvl = 256
        self.sym = True
        self.norm = True
        self.props = ['dissimilarity', 'correlation',
                      'homogeneity', 'contrast', 'ASM', 'energy']

    def extract(self):
        glcm = greycomatrix(self.img, distances=self.dists, angles=self.agls,
                            levels=self.lvl, symmetric=self.sym, normed=self.norm)
        feature = []
        glcm_props = [propery for name in self.props for propery in greycoprops(glcm, name)[
            0]]
        for item in glcm_props:
            feature.append(item)
        return feature


class GLCM2(object):
    def __init__(self, imageGrey, imageColour):
        self.img = imageGrey
        self.imgColour = imageColour
        (self.h, self.w) = imageGrey.shape[:2]
        self.dists = [5]
        self.agls = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.lvl = 256
        self.sym = True
        self.norm = True
        self.props = ['dissimilarity', 'correlation',
                      'homogeneity', 'contrast', 'ASM', 'energy']
        
    def glcmFeature(self,img):
        glcm = greycomatrix(img, distances=self.dists, angles=self.agls,
                            levels=self.lvl, symmetric=self.sym, normed=self.norm)
        feature = []
        glcm_props = [propery for name in self.props for propery in greycoprops(glcm, name)[
            0]]
        for item in glcm_props:
            feature.append(item)
        return feature

    def splitFour(image):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        topLeft = image[0:cY, 0:cX]
        topRight = image[0:cY, cX:w]
        bottomLeft = image[cY:h, 0:cX]
        bottomRight = image[cY:h, cX:w]

        return topLeft, topRight, bottomLeft, bottomRight

    def cropOrangeFeature(self):
        hsvImg = cv2.cvtColor(self.imgColour, cv2.COLOR_BGR2HSV)
        boundLower = np.array([0, 100, 45])
        boundUpper = np.array([225, 250, 255])
        maskOrange = cv2.inRange(hsvImg, boundLower, boundUpper)
        maskingData = np.invert(maskOrange)

        whitePtCoords = np.argwhere(maskingData)
        min_y = min(whitePtCoords[:, 0])
        min_x = min(whitePtCoords[:, 1])
        max_y = max(whitePtCoords[:, 0])
        max_x = max(whitePtCoords[:, 1])

        colourImageCrop = cv2.resize(self.imgColour[min_y:max_y, min_x:max_x], (576,216), interpolation = cv2.INTER_AREA)
        greyImageCrop = cv2.resize(self.img[min_y:max_y, min_x:max_x], (576,216), interpolation = cv2.INTER_AREA)
        binerImageCrop = cv2.resize(maskingData[min_y:max_y, min_x:max_x], (576,216), interpolation = cv2.INTER_AREA)
        return colourImageCrop, greyImageCrop, binerImageCrop

    def meanHSV(self, colourImg):
        hsvImg = cv2.cvtColor(colourImg, cv2.COLOR_BGR2HSV)
        averageH = np.average(hsvImg[0])
        averageS = np.average(hsvImg[1])
        averagev = np.average(hsvImg[2])
        return [averageH, averageS, averagev]
    
    def binerFeature(img):
        return np.sum(img)

    def extract(self):
        combineFeature = []
        featureHSV = []
        featureGLCM = []
        featureBiner = []
        
        colour, grey, biner = self.cropOrangeFeature()
        
        c1, c2, c3, c4 = self.splitFour(colour)
        g1, g2, g3, g4 = self.splitFour(grey)
        b1, b2, b3, b4 = self.splitFour(biner)
        
        
        featureHSV.append(self.meanHSV(c1))
        featureHSV.append(self.meanHSV(c2))
        featureHSV.append(self.meanHSV(c3))
        featureHSV.append(self.meanHSV(c4))
        
        featureGLCM.append(self.glcmFeature(g1))
        featureGLCM.append(self.glcmFeature(g2))
        featureGLCM.append(self.glcmFeature(g3))
        featureGLCM.append(self.glcmFeature(g4))
        
        featureBiner.append(self.binerFeature(b1))
        featureBiner.append(self.binerFeature(b2))
        featureBiner.append(self.binerFeature(b3))
        featureBiner.append(self.binerFeature(b4))
        
        combineFeature.append(featureGLCM)
        combineFeature.append(featureHSV)
        combineFeature.append(featureBiner)
        
        return combineFeature
        
        
