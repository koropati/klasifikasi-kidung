import argparse
import os
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputPath', required=True, help='Input Folder Data')
    ap.add_argument('-o', '--outputPath', required=True, help='Output Folder Data')
    args = ap.parse_args()
    
    createFolderDataset(args.inputPath, args.outputPath)

def createFolderDataset(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    for fileName in os.listdir(src):
        fullPathIn = os.path.join(src, fileName)
        classImage = fileName.split("_")
        folderClass = dst + "\\" + classImage[0]
        fullPathOut = folderClass + "\\" + fileName
        if not os.path.exists(folderClass):
            os.makedirs(folderClass)
        
        if os.path.exists(fullPathOut):
            os.remove(fullPathOut)
        print("COPY {} TO {}".format(fullPathIn, fullPathOut))
        shutil.copy(fullPathIn, fullPathOut)
    print("DONE COPY DATASET")
    
if __name__ == '__main__':
    main()