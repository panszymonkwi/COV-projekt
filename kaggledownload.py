import kaggle
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            print("Tworzenie folderu roboczego...")
            os.makedirs(directory)
    except OSError:
        print('Wystail blad')


createFolder('./archive8/')
os.chdir('./archive8/')

print("Pobieranie dataset'u...")
kaggle.api.authenticate()

kaggle.api.dataset_download_files('kasia12345/polish-traffic-signs-dataset', unzip=True)

print("Proces zakonczony")