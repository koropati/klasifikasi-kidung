import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops


class GLCM(object):
    def __init__(self, image, ):
        self.img = image
        self.dists=[5]
        self.agls=[0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.lvl=256
        self.sym=True
        self.norm=True
        self.props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    def extract(self):
        glcm = greycomatrix(self.img, distances=self.dists, angles=self.agls, levels=self.lvl, symmetric=self.sym, normed=self.norm)
        feature = []
        glcm_props = [propery for name in self.props for propery in greycoprops(glcm, name)[0]]
        for item in glcm_props:
            feature.append(item)
        return feature
