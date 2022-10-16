import argparse
import os
import sys

import pandas as pd
import numpy as np

from lib.csv import readCSVFloat, readCSVString
from sklearn.model_selection import train_test_split

# Command atau perintah untuk menjalankan file ini
#  python .\08_train_test_split.py -t 0.8 -i coba -s true -r 42 -o .\dataset-split
#  python .\08_train_test_split.py -t 0.8 -i coba -s true -r 42
# ------------------------------------
# -t adalah nilai presentase data train 0.8 = 80%
# -i adalah kode nama feture dataset yang ada pada folder "/dataset-generate-feature" karna akan meng load data dari ini.
# -s adalah parameter penentu stratify untuk splitting dataset pada modul "train_test_split" dari "sklearn"
# -r adalah parameter penentu random state untuk splitting dataset pada modul "train_test_split" dari "sklearn"
# -o adalah output folder untuk menyimpan hasil spliting dataset nya. bisa di kosongkan , kalau di kosngkan akan masuk ke folder "dataset-split"

# Starify Meaning in URL : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--train', type=float, required=True,
                    help='Input Nilai presentase train (ex: 0.8 untuk 80%)')
    ap.add_argument('-i', '--inputFeature', required=True,
                    help='Input Nama Feature dataset yang akan displit contoh : glcm, lbp, hog2 merujuk ke folder dataset-generate-feature')
    ap.add_argument('-s', '--stratify', type=bool, required=True,
                    help='Adalah nilai true atau false dari stratify method')
    ap.add_argument('-r', '--randomState', type=int, required=True,
                    help='Adalah nilai int dari random state method')
    ap.add_argument('-o', '--outputPath', default=".\\dataset-split",
                    help='Output Folder dataset yang sudah tersplit contoh /dataset-split')
    args = ap.parse_args()

    trainTestSplit(args.train, args.inputFeature, args.stratify, args.randomState, args.outputPath)


def trainTestSplit(train, inputFeature, stratify, randomState, outputPath):
    print("Processing...")
    # Path atau lokasi file feature dan label
    datasetFeaturePath = "dataset-generate-feature\\dataset_"+inputFeature+".csv"
    datasetLabelPath = "dataset-generate-feature\\dataset_"+inputFeature+"_label.csv"

    # Check Path atau lokasi file feature
    if not os.path.exists(datasetFeaturePath):
        print("File : {} NOT EXIST".format(datasetFeaturePath))
        sys.exit()

    # Check Path atau lokasi file label
    if not os.path.exists(datasetLabelPath):
        print("File : {} NOT EXIST".format(datasetLabelPath))
        sys.exit()

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    print("Loading dataset...")
    # dfFeature = pd.read_csv(datasetFeaturePath)
    dfFeature = readCSVFloat(datasetFeaturePath)
    # dfLabel = np.ravel(pd.read_csv(datasetLabelPath))
    dfLabel = np.ravel(readCSVString(datasetLabelPath))
    print("Done Loading dataset!")

    # Nama File Datset Train dan Test beserta labelnya
    nameFileDatasetTrain = "dataset_train_"+inputFeature+".csv"
    nameFileDatasetLabelTrain = "dataset_train_"+inputFeature+"_label.csv"
    nameFileDatasetTest = "dataset_test_"+inputFeature+".csv"
    nameFileDatasetLabelTest = "dataset_test_"+inputFeature+"_label.csv"

    # Mencari nilai presentase test ( nilai train itu 0 - 1, misal 0.8 jadi test nya 1 - 0.8 jadinya 0.2)
    testSize = 1 - train

    print("Spliting dataset...")
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            dfFeature, dfLabel, test_size=testSize, stratify=dfLabel, random_state=randomState)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            dfFeature, dfLabel, test_size=testSize, stratify=dfLabel, random_state=randomState)
    print("Done Splitting dataset!\n")

    print("INFO: --------------------------------")
    print("DATA X_TRAIN LENGTH: ", len(X_train))
    print("DATA Y_TRAIN LENGTH: ", len(y_train))
    print("=======================================")
    print("DATA X_TEST LENGTH: ", len(X_test))
    print("DATA Y_Test LENGTH: ", len(y_test))
    print("=======================================")

    print("Converting to dataframe...")
    # convert array into dataframe
    dataTrain = pd.DataFrame(X_train)
    labelDataTrain = pd.DataFrame(y_train)
    dataTest = pd.DataFrame(X_test)
    labelDataTest = pd.DataFrame(y_test)
    print("Done converting!\n")

    print("Saving...")
    
    # Pengecekan jika sudah ada file dengan nama tersebut maka akan di hapus dan di ganti dnegan data baru saat ini.
    if os.path.exists(outputPath+"\\"+nameFileDatasetTrain):
        os.remove(outputPath+"\\"+nameFileDatasetTrain)
    if os.path.exists(outputPath+"\\"+nameFileDatasetLabelTrain):
        os.remove(outputPath+"\\"+nameFileDatasetLabelTrain)
    if os.path.exists(outputPath+"\\"+nameFileDatasetTest):
        os.remove(outputPath+"\\"+nameFileDatasetTest)
    if os.path.exists(outputPath+"\\"+nameFileDatasetLabelTest):
        os.remove(outputPath+"\\"+nameFileDatasetLabelTest)
    
    # Simpan hasil Split
    dataTrain.to_csv(outputPath+"\\"+nameFileDatasetTrain, header=False, index=False)
    labelDataTrain.to_csv(outputPath+"\\"+nameFileDatasetLabelTrain, header=False, index=False)

    dataTest.to_csv(outputPath+"\\"+nameFileDatasetTest, header=False, index=False)
    labelDataTest.to_csv(outputPath+"\\"+nameFileDatasetLabelTest, header=False, index=False)
    print("Success Saving new dataset(splitted)")


if __name__ == '__main__':
    main()
