# klasifikasi-kidung
Core System Classification Traditional Balinese Song

# REQUIRED:
- git clone this repository
- go to folder repository using command line : ```cd klasifikasi-kidung```
- create your python env using command line : ```python -m venv myenv```
- activate your new env using command line : ```myenv\Scripts\activate```
- install depedency / library with this command line : ```pip install -r req.txt```

- Download Audio Dataset in Google Drive Link https://drive.google.com/drive/folders/1Crglr-xmXMMrL2mAEzydWRmM_BdvaZCq?usp=sharing 
- Download ```DatasetBaru.zip```
- Extract zip file ```DatasetBaru.zip```
- Create new folder called ```audio```
- COPY ALL AUDIO data from folder ```DatasetBaru``` to folder ```audio```

AND YOU CAN RUN ALL Script, Following step by steep with number of instruction below:

1. Copying all audio to folder dataset with Auto create class folder.
    ```python .\01_create_folder_dataset.py -i .\audio\ -o .\dataset\```

2. Create Image Spectogram from Audio Dataset Folder.
    h: 432
    w: 576
    ```python .\02_create_spectogram_dataset.py -i .\dataset\ -o .\dataset-spectogram\ -e png```

3. Generate Vector Feature by apply method ex: HOG, LBP or Both 
    * for HOG Feature:
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m hog```
    * for HOG2 Feature (from scikit image):
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m hog2```
    * for LBP Feature:
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m lbp```
    * for LBP2 Feature (from scikit image):
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m lbp2```
    * for combine (HOG+LBP):
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m combine```
    * for combine (HOG2+LBP2):
    ```python .\03_generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m combine2```

3. Create KFold Validation
    -f = kFold number
    -n = number of N Neighbors in KNN Classifier
    -o = distination of out folder to save excell report K-Fold Validation (optional) default : .\model-kfold-validation\
    -m = method extract feature (it is find file in folder .\dataset-generate-feature\ )
    ```python .\04_model_kfold_validation.py -f 5 -n 5 -m hog```