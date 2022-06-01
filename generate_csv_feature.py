import argparse
import os
from pydoc import classname
import shutil
import cv2

from lib.hog import HOG
from lib.lbp import LBP
from lib.csv import appendListAsRow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputPath', required=True,
                    help='Input Folder Dataset Image Spectogram')
    ap.add_argument('-o', '--outputPath', required=True,
                    help='Output Folder Dataset CSV')
    ap.add_argument('-m', '--methodFeature', required=True,
                    help='Metode of extracting Feature ex: (hog, lbp, combine)')
    args = ap.parse_args()

    generateCSVFeature(args.inputPath, args.outputPath, args.methodFeature)


def generateCSVFeature(src, dst, methodFeature):
    print("Generate Feature...")
    nameFileOut = dst + os.sep + "dataset_" + methodFeature + ".csv"
    nameFileLabelOut = dst + os.sep + "dataset_" + methodFeature + "_label.csv"
    if not os.path.exists(dst):
        os.makedirs(dst)

    if os.path.exists(nameFileOut):
        os.remove(nameFileOut)
    if os.path.exists(nameFileLabelOut):
        os.remove(nameFileLabelOut)

    for root, dirs, files in os.walk(src):
        for name in files:
            vectorFeature = []
            fullPathIn = root + os.sep + name
            classImage = name.split("_")[0]

            # img_in = cv2.imread(fullPathIn)
            img_gray = cv2.imread(fullPathIn, cv2.IMREAD_GRAYSCALE)

            if methodFeature == 'hog':
                myHOG = HOG(img_gray, 16, 8)
                hogVector, _ = myHOG.extract()
                vectorFeature.extend(hogVector)
            elif methodFeature == 'lbp':
                myLBP = LBP(img_gray)
                lbpVector = myLBP.extract()
                vectorFeature.extend(lbpVector)
            else:
                myHOG = HOG(img_gray, 16, 8)
                hogVector, _ = myHOG.extract()
                vectorFeature.extend(hogVector)
                myLBP = LBP(img_gray)
                lbpVector = myLBP.extract()
                vectorFeature.extend(lbpVector)
            print("Processed : {}".format(fullPathIn))
            appendListAsRow(nameFileOut, vectorFeature)
            appendListAsRow(nameFileLabelOut, [classImage])
    print("Done!")

if __name__ == '__main__':
    main()
