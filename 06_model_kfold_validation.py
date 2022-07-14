import argparse
import os, sys

import pandas as pd
import numpy as np
import xlsxwriter
import datetime

from lib.csv import readCSVFloat, readCSVString

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--kFold', required=True,
                    help='Input Folder Dataset Image Spectogram')
    ap.add_argument('-n', '--nNeighbors', required=True,
                    help='N Neighbors in KNN')
    ap.add_argument('-o', '--outputPath', default="model-kfold-validation",
                    help='Output Folder Report CSV')
    ap.add_argument('-m', '--methodFeature', required=True,
                    help='Metode of extracting Feature ex: (hog, lbp, combine)')
    args = ap.parse_args()

    modelValidation(args.kFold, args.nNeighbors, args.outputPath, args.methodFeature)

def modelValidation(kFold, nNeighbors, dst, methodFeature):
    print("Processing...")
    # Path atau lokasi file feature dan label
    datasetFeaturePath = "dataset-generate-feature\\dataset_"+methodFeature+".csv"
    datasetLabelPath = "dataset-generate-feature\\dataset_"+methodFeature+"_label.csv"
    
    # Check Path atau lokasi file feature
    if not os.path.exists(datasetFeaturePath):
        print("File : {} NOT EXIST".format(datasetFeaturePath))
        sys.exit()
    
    # Check Path atau lokasi file label
    if not os.path.exists(datasetLabelPath):
        print("File : {} NOT EXIST".format(datasetLabelPath))
        sys.exit()
        
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    # dfFeature = pd.read_csv(datasetFeaturePath)
    dfFeature = readCSVFloat(datasetFeaturePath)
    # dfLabel = np.ravel(pd.read_csv(datasetLabelPath))
    dfLabel = np.ravel(readCSVString(datasetLabelPath))
    
    knnCF = KNeighborsClassifier(n_neighbors=int(nNeighbors))
    
    cvScores = cross_val_score(knnCF, dfFeature, dfLabel, cv=int(kFold))
    
    currentDate = datetime.datetime.now()
    
    nameFile = "knn_"+nNeighbors+"_kfold_"+kFold+"_feature_"+methodFeature+"_Date_"+str(currentDate.day)+"_"+str(currentDate.month)+"_"+str(currentDate.year)+"_"+str(currentDate.hour)+"_"+str(currentDate.minute)+"_"+str(currentDate.second)+".xlsx"
    
    print("Creating Excel File...")
    workBook = xlsxwriter.Workbook(dst+"/"+nameFile)
    workSheet = workBook.add_worksheet()
    row = 0
    col = 0
    
    for item in (cvScores):
        workSheet.write(row, col, "Fold "+str(row+1))
        workSheet.write(row, col + 1, item)
        row += 1
    
    workSheet.write(row, col, "Mean")
    workSheet.write(row, col +1, np.mean(cvScores))
    workBook.close()
    print("Done!")

if __name__ == '__main__':
    main()