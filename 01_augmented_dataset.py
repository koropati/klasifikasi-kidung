import argparse
import os

from lib.augmented import Augmented

# python 01_augmented_dataset.py
# python .\01_augmented_dataset.py -i .\dataset-mentah\ -o .\dataset-with-augmented\


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputPath', required=True,
                    help='Input Folder Dataset Audio')
    ap.add_argument('-o', '--outputPath', required=True,
                    help='Output Folder Dataset Clean')
    args = ap.parse_args()

    augmentedDataset(args.inputPath, args.outputPath)


def augmentedDataset(src, dst):
    print("Running...")
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):

        for name in files:
            fullPathIn = root + os.sep + name
            print("Augmented Data : {}".format(fullPathIn))
            datasetAugmented = Augmented(fullPathIn, dst)
            datasetAugmented.create()

    print("Done!")


if __name__ == '__main__':
    main()
