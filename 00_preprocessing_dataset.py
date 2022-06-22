import argparse
import os

from lib.preprocessing import CleanAudio

# python 00_preprocessing_dataset.py 
# python .\00_preprocessing_dataset.py -i .\dataset-mentah\ -o .\audio\ -t 10 -d 5 -s 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputPath', required=True,
                    help='Input Folder Dataset Audio')
    ap.add_argument('-o', '--outputPath', required=True,
                    help='Output Folder Dataset Clean')
    ap.add_argument('-t', '--dbTreshold', required=True,
                    help='Ambang Batas dikatakan silent')
    ap.add_argument('-d', '--duration', required=True,
                    help='Durasi Potong (detik)')
    ap.add_argument('-s', '--siftDistance', required=True,
                    help='Selisih Pergeseran (detik)')
    args = ap.parse_args()

    cleanDataset(args.inputPath, args.outputPath, int(args.dbTreshold),
                 int(args.duration), int(args.siftDistance))


def cleanDataset(src, dst, dbTreshold, duration, siftDistance):
    print("Running...")
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):

        for name in files:
            fullPathIn = root + os.sep + name
            print("Preprocessing Data : {}".format(fullPathIn))
            datasetClean = CleanAudio(
                fullPathIn, dst, dbTreshold, duration, siftDistance)
            datasetClean.extract()
            
    print("Done!")


if __name__ == '__main__':
    main()
