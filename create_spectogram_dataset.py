import argparse
import os
import shutil

from lib.spectogram import CreateSpectogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputPath', required=True,
                    help='Input Folder Dataset Audio')
    ap.add_argument('-o', '--outputPath', required=True,
                    help='Output Folder Dataset Image Spectogram')
    ap.add_argument('-e', '--extensionOut', required=True,
                    help='Output Extension (jpg,png) Dataset Image Spectogram')
    args = ap.parse_args()

    createSpectogramDataset(args.inputPath, args.outputPath, args.extensionOut)


def createSpectogramDataset(src, dst, extension):
    print("Running...")
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        
        for name in files:
            fullPathIn = root + os.sep + name
            
            classImage = name.split("_")
            folderClass = dst + os.sep + classImage[0]
            nameOut = name.split(".")
            fullPathOut = folderClass + os.sep + nameOut[0] + "." + extension
            if not os.path.exists(folderClass):
                os.makedirs(folderClass)
        
            if os.path.exists(fullPathOut):
                os.remove(fullPathOut)
            
            
            mySpectogram = CreateSpectogram(fullPathIn, fullPathOut)
            mySpectogram.create()
            print("Created Image : {}".format(fullPathOut))
    print("Done!")

if __name__ == '__main__':
    main()
