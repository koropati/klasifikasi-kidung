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
- COPY ALL AUDIO data from ```DatasetBaru``` to folder ```audio```

AND YOU CAN RUN ALL Script, Following step by steep with number of instruction below:

1. Copying all audio to folder dataset with Auto create class folder.
    ```python .\create_folder_dataset.py -i .\audio\ -o .\dataset\```

2. Create Image Spectogram from Audio Dataset Folder.
    ```python .\create_spectogram_dataset.py -i .\dataset\ -o .\dataset-spectogram\ -e png```

3. Generate Vector Feature by apply method ex: HOG, LBP or Both 
    ```python .\generate_csv_feature.py -i .\dataset-spectogram\ -o .\dataset-generate-feature\ -m hog```