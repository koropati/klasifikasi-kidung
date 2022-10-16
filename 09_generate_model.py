import argparse
import os, sys
import pickle

import pandas as pd
import numpy as np
import xlsxwriter
import datetime

from lib.csv import readCSVFloat, readCSVString

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# How to run this file
# python .\09_generate_model.py -n 10 -i hog2 -o .\model-data -r .\model-report
# python .\09_generate_model.py -n 10 -i hog2
# --------------------------------------------
# -n adalah jumlah K ketetangaan dari model KNN
# -i adalah kode fitur dataset yang diambil dari folder "dataset-split" contoh hog2 artinya akan meload dataset_train_hog2.csv begitu juga dgn file label dan test nya
# -o adalah output model / folder untuk menyimpan model yang sudah dibuat dalam bentuk pickle ".pkl". Opsional, bisa di kosongkan. Jika di kosongkan maka akan mengambil default nilai nya yaitu ".\model-data"
# -r adalah destinasi folder untuk menyimpan hasil report dari klasifikasinya. Opsional. Bisa di kosongkan, jika di kosongkan maka defaultnya yaitu ".\model-report"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--nNeighbors',type=int, required=True,
                    help='N Neighbors in KNN')
    ap.add_argument('-o', '--outputPath', default=".\\model-data",
                    help='Output Folder Model yang sudah di buat')
    ap.add_argument('-i', '--inputDataset', required=True,
                    help='Input berupa kode nama dataset hasil split yang berada pada folder dataset-split contoh "hog"')
    ap.add_argument('-r', '--reportPath', default=".\\model-report",
                    help='Path folder untuk menyimpan hasil report dari model yang sudah dibuat berupa report akurasi.')
    args = ap.parse_args()

    generateModel(args.nNeighbors, args.outputPath, args.inputDataset, args.reportPath)

def create_classification_report(report, fileDestination):

    workBook = xlsxwriter.Workbook(fileDestination)
    workSheet = workBook.add_worksheet()
    row = 0
    col=0
    
    workSheet.write(row, col, "Class")
    workSheet.write(row, col+1, "Precision")
    workSheet.write(row, col+2, "Recall")
    workSheet.write(row, col+3, "F1 Score")
    workSheet.write(row, col+4, "Support")
    row += 1
    lines = report.split('\n')
    for line in lines[2:-3]:
        
        row_data = line.split('      ')
        workSheet.write(row, col, row_data[0])
        workSheet.write(row, col+1, row_data[1])
        workSheet.write(row, col+2, row_data[2])
        workSheet.write(row, col+3, row_data[3])
        workSheet.write(row, col+4, row_data[4])
        row += 1
        
    workBook.close()
    
def generateModel(nNeighbors, outputPath, inputDataset, reportPath):
    print("Processing...")
    # Path atau lokasi file feature dan label
    
    datasetTrainPath = "dataset-split\\dataset_train_"+inputDataset+".csv"
    datasetTrainLabelPath = "dataset-split\\dataset_train_"+inputDataset+"_label.csv"
    datasetTestPath = "dataset-split\\dataset_test_"+inputDataset+".csv"
    datasetTestLabelPath = "dataset-split\\dataset_test_"+inputDataset+"_label.csv"
    
    # Check Path atau lokasi file feature
    if not os.path.exists(datasetTrainPath):
        print("File : {} NOT EXIST".format(datasetTrainPath))
        sys.exit()
    
    # Check Path atau lokasi file label
    if not os.path.exists(datasetTrainLabelPath):
        print("File : {} NOT EXIST".format(datasetTrainLabelPath))
        sys.exit()
    
    # Check Path atau lokasi file feature
    if not os.path.exists(datasetTestPath):
        print("File : {} NOT EXIST".format(datasetTestPath))
        sys.exit()
    
    # Check Path atau lokasi file label
    if not os.path.exists(datasetTestLabelPath):
        print("File : {} NOT EXIST".format(datasetTestLabelPath))
        sys.exit()
    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        
    if not os.path.exists(reportPath):
        os.makedirs(reportPath)
    
    print("Reading Dataset...")
    # Read CSV File before create model
    fiturTrain = readCSVFloat(datasetTrainPath)
    labelTrain = np.ravel(readCSVString(datasetTrainLabelPath))
    
    fiturTest = readCSVFloat(datasetTestPath)
    labelTest = np.ravel(readCSVString(datasetTestLabelPath))
    
    print("Creating Model...")
    # Create Model Here
    knnCF = KNeighborsClassifier(n_neighbors=int(nNeighbors))
    knnCF.fit(fiturTrain,labelTrain)
    
    print("Saving Model...")
    # Simpan Data Model
    # create destinasi + nama model yang akan di simpan
    pathDataModel = outputPath + "model_fitur_" + inputDataset + ".pkl"
    # Check jika sudah ada file tersebut maka di hapus dan di create yang baru
    if os.path.exists(pathDataModel):
        os.remove(pathDataModel)
    # Its important to use binary mode 
    knnPickle = open(pathDataModel, 'wb') 
    # source, destination 
    pickle.dump(knnCF, knnPickle)  
    # close the file
    knnPickle.close()
    # End Simpan Data Model
    print("Done saving model!")
    
    print("Creating Report...")
    # Start Membuat Report
    # Klasifikasi data dari data test
    hasilPrediksiTest = knnCF.predict(fiturTest)
    skorNilaiTest = knnCF.score(fiturTest, labelTest) #hasilnya berupa nilai dari 0 sampai 1 example: 0.88 adalah 88%
    percentSkorNilaiTest = skorNilaiTest * 100
    
    # Mencari data Confusion matrix
    dataConfusionMatrix = confusion_matrix(labelTest, hasilPrediksiTest)
    classificationReport = classification_report(labelTest, hasilPrediksiTest)
    
    # convert array into dataframe
    dfConfusionMatrix = pd.DataFrame(dataConfusionMatrix)
    # dfclassificationReport = pd.DataFrame(classificationReport)
    
    print("|REPORT MODEL| -----------------------------------")
    print(" -> Skor Test   : {}".format(skorNilaiTest))
    print(" -> Skor Test % : {}".format(percentSkorNilaiTest))
    print("|------------------------------------------------|")
    print("|Classification Report |-------------------------|")
    print(classificationReport)
    
    
    # Simpan Model Report
    print("Creating Report Excel File...")
    currentDate = datetime.datetime.now()
    nameFile = "report_model_"+inputDataset+"_Date_"+str(currentDate.day)+"_"+str(currentDate.month)+"_"+str(currentDate.year)+"_"+str(currentDate.hour)+"_"+str(currentDate.minute)+"_"+str(currentDate.second)+".xlsx"
    nameFileReport = "report_classification_model_"+inputDataset+"_Date_"+str(currentDate.day)+"_"+str(currentDate.month)+"_"+str(currentDate.year)+"_"+str(currentDate.hour)+"_"+str(currentDate.minute)+"_"+str(currentDate.second)+".xlsx"
    nameFileReportConfusionMatrix = "report_confusion_matrix_model_"+inputDataset+"_Date_"+str(currentDate.day)+"_"+str(currentDate.month)+"_"+str(currentDate.year)+"_"+str(currentDate.hour)+"_"+str(currentDate.minute)+"_"+str(currentDate.second)+".xlsx"
    
    # create_classification_report(classificationReport, reportPath+"/"+nameFileReport)
    dfConfusionMatrix.to_csv(reportPath+"/"+nameFileReportConfusionMatrix, header=False, index=False)
    
    workBook = xlsxwriter.Workbook(reportPath+"/"+nameFile)
    workSheet = workBook.add_worksheet()


    workSheet.write(0, 0, "Nilai K pada KNN")
    workSheet.write(0, 1, nNeighbors)
    
    workSheet.write(1, 0, "Skor Test")
    workSheet.write(1, 1, skorNilaiTest)
    
    workSheet.write(2, 0, "Skor Test %")
    workSheet.write(2, 1, percentSkorNilaiTest)
    
    workBook.close()
    print("Done!")

if __name__ == '__main__':
    main()