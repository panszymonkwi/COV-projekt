print("*******************************************************")
print("\n P R O J E K T")
print(" Detekcja i klasyfikacja znaków drogowych na zdjęciach.")
print(" autor : Szymon Kwidzinski")
print("\n*******************************************************")
print("\n 1. Wczytywanie bibliotek")
print("pip install pytorch-lightning przed uruchomieniem kodu")
print("      - import pytorch-lightning")
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

print("      - import numpy")
import numpy as np

print("      - import matplotlib")
import matplotlib.pyplot as plt

print("      - import cv2")
import cv2

print("      - import torch")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
from torch.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear, MaxPool2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

print("      - import os")
import os
from os import listdir

print("      - import torchvision")
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder

print("      - import PIL")
import PIL


print(" 2. Wybor modelu sieci do klasyfikacji znakow drogowych")
print("    2.1 nowo utworzony model sieci konwolucyjnej ModelTSNet (okrojona siec VGG16)")
print("    2.2 model wstepnie wytrenowany models.resnet18(pretrained=True)")
print("    2.3 model wstepnie wytrenowany models.squeezenet1_1(pretrained=True)")
print(" 3. Przygotowanie zbioru danych uczących model")
print("    3.1 Rozpakowanie zbioru danych z 92 klasami znaków z Kaggle'a")
print("    3.2 Zdefiniowanie klasy TrafficSignData")
print(" 4. Przygotowanie zbioru nazw powyższych znakow SignTnames.csv")
print(" 5. Trening modelu sieci konwolucyjnej na zbiorze danych")
print(" 6. Przeglad wynikow treningu na TensorBoard")
print(" 7. Wizualizacja transformacji danych")
print(" 8. Wizualizacja detekcji przy pomocy 'cascade.xml' ")
print("    8.1 Wizualizacja detekcji i klasyfikacji pojedynczego obrazka")
print("    8.2 Wizualizacja detekcji i klasyfikacji całego folderu Input")
print(" 9. Demo zapisanych efektów")
print("10. Podsumowanie")
print(" ")


def kontynuacja():
    char = " "
    while char != "n" and char != "N" and char != "t" and char != "T":
        char = input("Kontynuowac? (t,n): ")
    if char == "n" or char == "N":
        kontynuacja = False
    else:
        kontynuacja = True
    return kontynuacja


if not kontynuacja():
    os._exit(0)


print("\n******** W Y B O R  M O D E L U  S I E C I  D O  K L A S Y F I K A C J I  Z N A K O W  ********\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pracuje na {device}")  # przejście 1 epoki na CPU zajmuje około 2600 sekund, na GPU 66 sekund
print("\nDOSTEPNE MODELE SIECI:")
print("   1. ModelTSN zdefiniowany od podstaw (okrojone VGG16)")
print("   2. ModelRS na bazie ResNet-18 (pretrained)")
print("   3. ModelSN na bazie SqueezeNet 1.1 (pretrained)")

wybor = " "
while wybor != "1" and wybor != "2" and wybor != "3":
    wybor = input("\nWybierz model (1,2,3): ")
if wybor == "1":
    from ModelTSN import *
    modelTSN = ModelTSNet()
    torch.save(modelTSN, "modelTSN.h5")
    wybrany_model = modelTSN
    print(wybrany_model)

elif wybor == "2":
    from ModelRS import *
    modelR18 = ModelRS()
    torch.save(modelR18, "modelRS.h5")
    wybrany_model = modelR18
    print(wybrany_model.backbone)

else:
    from ModelSN import *
    modelSN = ModelSN()
    torch.save(modelSN, "modelSN.h5")
    wybrany_model = modelSN
    print(wybrany_model.backbone)

wybrany_model = wybrany_model.to(device)  # model przenosiony na aktualnie dostępne urządzenie

print("\n********* P R Z Y G O T O W A N I E  D A N Y C H  U C Z A C Y C H **********\n")

#!unzip archive8.zip

# Load Traffic Sign data
train_dataset = ImageFolder(root="archive8/train/")
test_dataset = ImageFolder(root="archive8/test/")


class TrafficSignData(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0), (1)),
            ]
        )
        self.train_dataset = ImageFolder(root="archive8/train/", transform=transform)
        print("train_dataset = ImageFolder(root='archive8/train/')")
        print("Rozmiar zbioru train_dataset:", len(self.train_dataset))

        self.test_dataset = ImageFolder(root="archive8/test/", transform=transform)
        print("test_dataset = ImageFolder(root='archive8/test/')")
        print("Rozmiar zbioru test_dataset:", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

daneTS = TrafficSignData()
daneTS.setup()

print("\n***** W C Z Y T A N I E  U T W O R Z O N E J  T A B E L I  N A Z W  Z N A K O W *****\n")

num = int(input("Ile nazw znakow wyswietlic? (0-92): "))
if num > 92:
    num = 92
if num < 0:
    num = 0
print("Wpisano:", num)

# Otwarcie tabeli nazw znaków drogowych
file = open("signTnames.csv")
labelTSnames = file.read().strip().split("\n")[1:]
labelnames = [lab.split(",")[1] for lab in labelTSnames]
for i in range(0, num):
    print(labelnames[i])
# Przegląd nazw znaków drogowych zbioru danych
for i in range(3):
    image, label = train_dataset[i * 200]
    print("\nD E M O  ", i + 1)
    print("Nr znaku : ", label)
    labelname = labelTSnames[label].split(",")[1]
    print("Znak : ", labelname)
    print(np.shape(image))
    plt.figure(num=i + 1, figsize=(6, 4))
    plt.imshow(image)
    plt.show()

print("\n***** T R E N I N G  M O D E L U  S I E C I  N A  Z B I O R Z E  D A N Y C H  *****\n")
# Trening modelu sieci konwolucyjnej na zbiorze danych

logger = TensorBoardLogger("lightning_logs", name="model")
trainer = pl.Trainer(logger=logger, max_epochs=5, log_every_n_steps=1)

char = " "
while char != "n" and char != "N" and char != "t" and char != "T":
    char = input("\nOminac trening modelu? (t,n): ")

if char == "n" or char == "N":
    trainer.fit(wybrany_model, daneTS)
    if wybor == "1":
        torch.save(wybrany_model, "modelTSN.h5")
    if wybor == "2":
        torch.save(wybrany_model, "modelRS.h5")
    if wybor == "3":
        torch.save(wybrany_model, "modelSN.h5")
    print("Trening")
else:
    if wybor == "1":
        modelTSN = torch.load("modelTSN.h5")
    if wybor == "2":
        modelR18 = torch.load("modelRS.h5")
    if wybor == "3":
        modelSN = torch.load("modelSN.h5")

# print("\n********* P O D S U M O W A N I E  W Y N I K O W  T R E N I N G U **********\n")
# Podsumowanie wyników treningu
#%load_ext tensorboard
# tensorboard --logdir "lighting_logs/"

print("\n********* F U N K C J A  W I Z U A L I Z A C J I  T R A N S F O R M A C J I **********\n")
# definicja transformacji

transform = transforms.Compose(
    [
        #transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0), (1)),
    ]
)

# funkcja wizualizacji transformacji
def visualize_tsd(dataloader, od, do):
    iterable = iter(dataloader)
    for i in range(od):
        input_tensor = next(iterable)[0]
    for i in range(od, do):
        with torch.no_grad():
            input_tensor = next(iterable)[0]
            transformed_tensor = transform(input_tensor).permute(1, 2, 0)
            transformed_tensor = transformed_tensor.to(device)
            transformed_grid = torchvision.utils.make_grid(transformed_tensor)
            transformed_tensor = transformed_tensor.cpu()
            print(i, input_tensor)
            # Plot the results side-by-side
            fig = plt.figure(num=i + 1, figsize=(8, 4))
            ax = fig.subplots(1, 2)
            ax[0].imshow(input_tensor)
            ax[0].set_title("Dataset Images")
            ax[0].axis("off")

            ax[1].imshow(transformed_grid)
            ax[1].set_title("Transformed Images")
            ax[1].axis("off")
            plt.show()


visualize_tsd(train_dataset, 0, 3)


# print("\n********* F U N K C J A  W I Z U A L I Z A C J I  D E T E K C J I  I  K L A S Y F I K A C J I**********\n")

# Rozpakowanie zbiorów pokazowych
#!unzip test-images.zip

# funkcja wizualizacji detekcji przy pomocy cascady

cascade = cv2.CascadeClassifier("cascade.xml")
model = wybrany_model
labelnames = [lab.split(",")[1] for lab in labelTSnames]


def visualize_detect(path, namefile):
    print(namefile)
    img = cv2.imread(path + namefile)
    if img is None:
        print("Nie znaleziono pliku ", namefile)
    else:
        arr_image = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(namefile)
        plt.show()
        try:
            print(img.shape)
        except AttributeError:
            print("shape not found")
        # detection
        #cv2.ocl.setUseOpenCL(False)
        img_out = cv2.imread(path + namefile)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        boxes = cascade.detectMultiScale(img_out,scaleFactor=1.01,minNeighbors=7,minSize=(24, 24),maxSize=(224, 224))
        print("Liczba wykrytych znaków: ", len(boxes))
        if len(boxes) > 0:
            # recognition and drawing boundary boxes on input image
            for (x, y, w, h) in boxes:
                print("\nbox: ", x, y, w, h)
                img_rect = cv2.rectangle(img_out, (x, y), (x + w, y + h), (255, 255, 0), 2)

                cropped_image = arr_image[y : y + h, x : x + w, :]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = PIL.Image.fromarray(cropped_image)
                cropped_image = transform(cropped_image)
                cropped_image1 = cropped_image.permute(1, 2, 0)
                cropped_image = np.expand_dims(cropped_image, axis=0)

                cropped_image = torch.from_numpy(cropped_image)

                cropped_image = model(cropped_image)
                preds = F.softmax(cropped_image, dim=1)
                print("predictions: ", preds.detach().numpy())
                label = preds.argmax(axis=1)[0]
                label = label.item()
                value = preds.max()
                if value > 0.01:
                    labname = labelnames[label]
                    labnameS = labelnames[label].split(" ")[0]
                    print(f"Nr znaku: {label}, max pred: {value:.3f}, znak: {labname}")
                    cv2.putText(
                        img_out,
                        labnameS,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (255, 0, 0),
                        2,
                    )
                else:
                    print(
                        namefile,
                        f", max pred: {value:.3f} zbyt małe prawdopodobieństwo predykcji",
                    )

                plt.figure(num=i, figsize=(4, 4))
                plt.title("Wykryty znak")
                plt.imshow(cropped_image1)
                plt.show()
        plt.figure(num=i, figsize=(6, 4))
        plt.imshow(img_out)
        plt.title("Zlokalizowane znaki")
        plt.show()
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test-images/Output/"+ wybor + "-" + namefile, img_out)


print(
    "\n*********   W I Z U A L I Z A C J A  D E T E K C J I  I  K L A S Y F I K A C J I  O B R A Z K A  **********\n"
)
# wizualizacja detekcji pojedynczego obrazka

namefile = "input00.jpg"
path = ""  # 'test-images/Input/'
i = 0
plt.figure(num=i, figsize=(6, 4))
visualize_detect(path, namefile)

print(
    "\n*********  W I Z U A L I Z A C J A  D E T E K C J I  I  K L A S Y F I K A C J I  F O L D E R U  I N P U T **********\n"
)
if kontynuacja():

    # wizualizacja detekcji całego folderu Input
    path = "test-images/" #""
    subpath =  "Input/" #"widoki/"
    print("Liczba obrazów:", len(os.listdir(path + subpath)))
    i = 0
    for image in sorted(os.listdir(path + subpath)):
        i = i + 1
        print(f"\n{i}.")
        plt.figure(num=i, figsize=(6, 4))
        visualize_detect(path + subpath, image)


print(
    "\n*********  D E M O   Z A P I S A N Y C H   E F E K T O W  **********\n"
)
# przegląd zapisanych efektów detekcji

if kontynuacja():

    path = "test-images/Output/"
    print("Liczba obrazów:", len(os.listdir(path)))
    i = 0
    for namefile in sorted(os.listdir(path)):
        i = i + 1
        print(f"{i}.  {namefile}")
        image = cv2.imread(path + namefile)
        if image is None:
            print("Nie znaleziono pliku ", namefile)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(num=i, figsize=(6, 4))
            plt.title("Przeglad efektow")
            plt.imshow(img)
            plt.show()


print("\n******************** KONIEC ********************\n")
