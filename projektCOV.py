print("*******************************************************")
print("\n P R O J E K T")
print(" Detekcja i klasyfikacja znaków drogowych na zdjęciach.")
print(" autor : Szymon Kwidzinski")
print("\n*******************************************************")
print("\n 1. Wczytywanie bibliotek")
print("pip install pytorch-lightning przed uruchomieniem kodu")
print("      - import pytorch-lightning")  
import pytorch_lightning as pl
print("      - import numpy")
import numpy as np
print("      - import matplotlib")
import matplotlib.pyplot as plt
print("      - import cv2")
import cv2
print("      - import torch")
import torch 
from torch import optim
from torch.optim import Adam
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, Linear, Dropout
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
print("      - import os")
import os
from os import listdir
print("      - import torchvision")
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
print("      - import PIL")
import PIL

writer = SummaryWriter()

print(" 2. Utworzenie modeli sieci do klasyfikacji znaków")
print("    2.a nowy model sieci konwolucyjnej ModelTSNet")
print("    2.b model wytrenowany models.vgg16(pretrained=True)")
print(" 3. Przygotowanie zbioru danych uczących model")
print("    3.1 Rozpakowanie zbioru danych z 92 klasami znaków")
print("    3.2 Utworzenie klasy TrafficSignData")
print(" 4. Przygotowanie zbioru nazw powyższych znakow")
print(" 5. Trening modelu sieci konwolucyjnej na zbiorze danych")
print(" 6. Podsumowanie wynikow treningu")
print(" 7. Wizualizacja transformacji danych")
print(" 8. Wizualizacja detekcji przy pomocy cascady")
print("    8.1 Wizualizacja detekcji pojedynczego obrazka")
print("    8.2 Wizualizacja detekcji całego folderu Input")
print(" 9. Demo zapisanych efektów detekcji")
print("10. Podsumowanie")
print(" ")

def kontynuacja():
  char = ' '
  while (char != 'n' and char != 'N' and char != 't' and char != 'T'):
    char = input("Kontynuowac? (t,n): ")
  if (char == 'n' or char == 'N'):
    kontynuacja = False
  else: kontynuacja = True
  return  kontynuacja

if (not kontynuacja()):
  os._exit(0)

# Inicjalizacja loggera TensorBoard, logi zapisujemy do folderu 'logs'
logs_path = 'logs/'
tensorboard_logger = SummaryWriter('logs/')
#tensorboard_logger = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
#tensorboard_initialized = True


print("\n**** U t w o r z e n i e   m o d e l u   s i e c i   d o   k l a s y f i k a c j i   z n a k ó w  *****\n")
class ModelTSNet(pl.LightningModule):
  def __init__(self, num_classes=92):
    super().__init__()
    self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # [3, 224, 224]
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) # [64, 224, 224]
    self.bnorm1 = BatchNorm2d(64)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  #[64, 112, 112]
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) #[128, 112, 112]
    self.bnorm2 = BatchNorm2d(128)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) #[128, 56, 56]
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) #[256, 56, 56]
    self.bnorm3 = BatchNorm2d(256)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) #[256,28, 28]
    self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[512, 28, 28]
    self.bnorm4 = BatchNorm2d(512)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.linear1 = nn.Linear(512*14*14, 1024)
    self.drop1 = nn.Dropout(0.5)
    self.linear2 = nn.Linear(1024, 92)

    self.loss_function = nn.CrossEntropyLoss()
    self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    self.train_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass", average='macro')
    self.val_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass", average='macro')

  def forward(self, x):
    with torch.no_grad():
      # the first conv group
      x = F.relu(self.conv1_1(x))
      x = F.relu(self.conv1_2(x))
      x = self.bnorm1(x)
      x = self.maxpool1(x)
      # the second conv group
      x = F.relu(self.conv2_1(x))
      x = F.relu(self.conv2_2(x))
      x = self.bnorm2(x)
      x = self.maxpool2(x)
      # the third conv group
      x = F.relu(self.conv3_1(x))
      x = F.relu(self.conv3_2(x))
      x = self.bnorm3(x)
      x = self.maxpool3(x)
      # the fourth conv group
      x = F.relu(self.conv4_1(x))
      x = F.relu(self.conv4_2(x))
      x = self.bnorm4(x)
      x = self.maxpool4(x)
      # flatten
      x = x.reshape( x.shape[0], -1)
      # the first linear layer with ReLU
      x = self.linear1(x)
      x = F.relu(x)
      # the first dropout
      x = self.drop1(x)
      # the second linear layer with sofmax
    x = self.linear2(x)
    return x

  # Inicjalizacja loggera TensorBoard, logi zapisujemy do folderu 'logs'
   # tensorboard_logger = SummaryWriter('logs/')
    #tensorboard_initialized = False


  def configure_optimizers(self):
    optimizer =  optim.Adam(self.parameters(), lr = 0.01)
    return optimizer

  def training_step(self, train_batch, batch_idx):

    inputs, labels = train_batch
    outputs = self.forward(inputs.float())

    loss = self.loss_function(outputs, labels)
    #if (not (tensorboard_initialized)):
                    # Inicjalizacja grafu odbywa się tylko raz, kolekcja data_inputs się nie zmienia,
                    # nie ma potrzeby inicjalizować ponownie
    #writer.add_scalar("Loss/train", loss, 1)
    #tensorboard_logger.add_graph(self, train_batch)
                    #tensorboard_initialized = True
    
    #print(loss)
    self.log('train_loss', loss, on_step= True, on_epoch = True)
    outputs = F.softmax(outputs, dim =1)

    #label1 = outputs.argmax( axis=1)[0]
    #label1 = label1.item()
    #print(f"znak  {labels[i]} , {label1}")

    self.train_acc(outputs, labels)
    self.log('train_acc', self.train_acc, on_epoch=True, on_step= False)
    self.train_macro_f1(outputs, labels)
    self.log('train_macro_f1', self.train_macro_f1, on_epoch=True, on_step= False)
    return loss

  def validation_step(self, val_batch, batch_idx):

    inputs, labels = val_batch
    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)
    self.log('val_loss', loss,  on_step= True, on_epoch = True)

    #writer.add_scalar("Loss/val", loss, 1)
    #tensorboard_logger.add_graph(self, val_batch)

    outputs = F.softmax(outputs, dim =1)

    #label1 = outputs.argmax( axis=1)[0]
    #label1 = label1.item()
    #print(f"test znak {labels[i]} , {label1}")

    self.val_acc(outputs, labels)
    self.log('val_acc', self.val_acc, on_epoch=True, on_step= False)
    self.val_macro_f1(outputs, labels)
    self.log('val_macro_f1', self.val_macro_f1, on_epoch=True, on_step= False)
    return loss
writer.flush()
#tensorboard_logger.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Pracuje na {device}')  # przejście 1 epoki na CPU zajmuje około 2600 sekund, na GPU 66 sekund

modelTSN = ModelTSNet()
print(modelTSN)

torch.save(modelTSN, 'modelTSN.h5')
modelTSN = modelTSN.to(device) # model przenosiony na aktualnie dostępne urządzenie


modelvgg16 = models.vgg16(pretrained=True)
modelvgg16.features

'''
class VGG16(pl.LightningModule):
  def __init__(self, num_classes= 92):
    super().__init__()

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.backbone = modelvgg16.features

    self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   # TODO

    # 2 warstwy liniowe - warstwa ukryta - 500 neuronów
    self.fc1 = nn.Linear(4608, 500)        # TODO
    self.fc2 = nn.Linear(500, num_classes) # TODO

  def forward(self, x):
    
    x = F.relu(self.conv1_1(x))
    x = F.relu(self.conv1_2(x))
    x = self.maxpool(x)
    x = F.relu(self.conv2_1(x))
    x = F.relu(self.conv2_2(x))
    x = self.maxpool(x)
    x = F.relu(self.conv3_1(x))
    x = F.relu(self.conv3_2(x))
    x = F.relu(self.conv3_3(x))
    x = self.maxpool(x)
    x = F.relu(self.conv4_1(x))
    x = F.relu(self.conv4_2(x))
    x = F.relu(self.conv4_3(x))
    x = self.maxpool(x)
    x = F.relu(self.conv5_1(x))
    x = F.relu(self.conv5_2(x))
    x = F.relu(self.conv5_3(x))
    x = self.maxpool(x)
    x = x.reshape(x.shape[0], -1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, 0.5)
    x = F.relu(self.fc2(x))
    x = F.dropout(x, 0.5)
    x = self.fc3(x)
    return x

  def configure_optimizers(self):
    optimizer =  optim.SGD(self.parameters(), lr = 0.01)
    return optimizer

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch


    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)

    self.log('train_loss', loss, on_step= True, on_epoch = True)

    outputs = F.softmax(outputs, dim =1)

    self.train_acc(outputs, labels)
    self.log('train_acc', self.train_acc, on_epoch=True, on_step= False)

    self.train_macro_f1(outputs, labels)
    self.log('train_macro_f1', self.train_macro_f1, on_epoch=True, on_step= False)


    return loss

  def validation_step(self, val_batch, batch_idx):
    inputs, labels = val_batch


    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)

    self.log('val_loss', loss,  on_step= True, on_epoch = True)


    outputs = F.softmax(outputs, dim =1)

    self.val_acc(outputs, labels)
    self.log('val_acc', self.val_acc, on_epoch=True, on_step= False)

    self.val_macro_f1(outputs, labels)
    self.log('val_macro_f1', self.val_macro_f1, on_epoch=True, on_step= False)

    return loss


modelVGG = VGG16()
'''

modelSN = models.squeezenet1_1(pretrained=True)
class ModelSN(pl.LightningModule):
  def __init__(self, num_classes=92):
    super().__init__()

    self.backbone = modelSN.features
    self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    self.fc = nn.Linear(512, 92)

    self.loss_function = nn.CrossEntropyLoss()

    self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
    self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)

    self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
    self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)

    self.train_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass", average='macro')
    self.val_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, task="multiclass", average='macro')

  def forward(self, x):
      self.backbone.eval()
      with torch.no_grad():
          x = self.backbone(x)
          x = self.pooling(x).flatten(1)
      x = self.fc(x)
      return x

  def configure_optimizers(self):
    optimizer =  optim.Adam(self.parameters(), lr = 0.0001)
    return optimizer

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch

    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)

    self.log('train_loss', loss, on_step= True, on_epoch = True)

    outputs = F.softmax(outputs, dim =1)

    self.train_accuracy(outputs, labels)
    self.log('train_accuracy', self.train_accuracy, on_epoch=True, on_step= False)

    self.train_precision(outputs, labels)
    self.log('train_precision', self.train_precision, on_epoch=True, on_step= False)

    self.train_recall(outputs, labels)
    self.log('train_recall', self.train_recall, on_epoch=True, on_step= False)

    self.train_macro_f1(outputs, labels)
    self.log('train_macro_f1', self.train_macro_f1, on_epoch=True, on_step= False)

    #target = outputs.round().long()
    #tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='multilabel', threshold=0.5)

    #self.train_roc_curve = metrics.roc_auc_score(tp,fp)
    #self.train_roc_curve(outputs, labels)
    #self.log('train_roc_curve', self.train_roc_curve, on_epoch=True, on_step= False)

    return loss

  def validation_step(self, val_batch, batch_idx):
    inputs, labels = val_batch

    outputs = self.forward(inputs.float())
    loss = self.loss_function(outputs, labels)

    self.log('val_loss', loss,  on_step= True, on_epoch = True)

    outputs = F.softmax(outputs, dim =1)

    self.val_accuracy(outputs, labels)
    self.log('val_accuracy', self.val_accuracy, on_epoch=True, on_step= False)

    self.val_precision(outputs, labels)
    self.log('val_precision', self.val_precision, on_epoch=True, on_step= False)

    self.val_recall(outputs, labels)
    self.log('val_recall', self.val_recall, on_epoch=True, on_step= False)

    self.val_macro_f1(outputs, labels)
    self.log('val_macro_f1', self.val_macro_f1, on_epoch=True, on_step= False)

    #target = outputs.round().long()
    #tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='multilabel', threshold=0.5)

    #self.val_roc_curve = metrics.roc_auc_score(tp,fp)
    #self.train_roc_curve(outputs, labels)
    #self.log('train_roc_curve', self.train_roc_curve, on_epoch=True, on_step= False)

    return loss

modelSN = ModelSN()
print("\n********* P R Z Y G O T O W A N I E  D A N Y C H  U C Z A C Y C H **********\n")

#!unzip archive8.zip

# Load Traffic Sign data
train_dataset = ImageFolder(root='archive8/train/')
#print("train_dataset = ImageFolder(root='archive8/train/')")
test_dataset = ImageFolder(root='archive8/test/')
#print("test_dataset = ImageFolder(root='archive8/test/')")

class TrafficSignData(pl.LightningDataModule):
  def __init__(self, batch_size = 32):
    super().__init__()
    self.batch_size = batch_size
  def setup(self, stage=None ):
    transform = transforms.Compose([
                                    #transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0), (1))
                                ])
    self.train_dataset = ImageFolder(root='archive8/train/', transform=transform)
    print("train_dataset = ImageFolder(root='archive8/train/')")
    #train_dataset = self.train_dataset
    print("Rozmiar zbioru train_dataset:" , len(self.train_dataset))

    #self.image_train,self.label_train = self.train_dataset
    #self.image_test,self.label_test = self.test_dataset

    #train_n=int(0.9*len(self.train_dataset))
    #self.train_dataset, self.valid_dataset = np.split(self.train_dataset, [train_n])
    #self_label_train, self_label_test = np.split(self_label_train, [train_n])

    # self.valid_dataset = ImageFolder(root='valid/', transform=transform)
    #valid_dataset = self.valid_dataset
    #print(len(self.valid_dataset))
    self.test_dataset = ImageFolder(root='archive8/test/', transform=transform)
    print("test_dataset = ImageFolder(root='archive8/test/')")
    #test_dataset = self.test_dataset
    print("Rozmiar zbioru test_dataset:" , len(self.test_dataset))

    #for label in self.label_test :
    #    print (label)

  def train_dataloader(self):
    return  DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
  def val_dataloader(self):
    return  DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
  #def test_dataloader(self):
  #  return  DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)

daneTS = TrafficSignData()
daneTS.setup()

print("\n***** W C Z Y T A N I E  U T W O R Z O N E J  T A B E L I  N A Z W  Z N A K O W *****\n")

num = int(input("Ile nazw znakow wyswietlic (0-92): "))
if (num > 92):
  num = 92
if (num < 0):
  num = 0
print(num, " ", type(num))

#Otwarcie tabeli nazw znaków drogowych
file = open("signTnames.csv")
labelTSnames = file.read().strip().split("\n")[1:]
labelnames = [lab.split(",")[1] for lab in labelTSnames]
for i in range(0,num):
    print (labelnames[i])
#Przegląd nazw znaków drogowych zbioru danych
for i in range(3):
  image, label = train_dataset[i*200]
  print("\nD E M O  ", i+1)
  print("Nr znaku : ", label)
  labelname = labelTSnames[label].split(",")[1]
  print("Znak : ", labelname)
  print(np.shape(image))
  plt.figure(num=i+1, figsize=(6,4))
  plt.imshow(image)
  plt.show()
#from PIL import Image
#image = Image.open("input01.jpg")
#image.show()

print("\n***** T R E N I N G  M O D E L U  S I E C I  N A  Z B I O R Z E  D A N Y C H  *****\n")
#Trening modelu sieci konwolucyjnej na zbiorze danych
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs", name="modelTSN")
trainer = pl.Trainer(logger = logger, max_epochs = 1, log_every_n_steps =1)

writer.close()

char = ' '
while (char != 'n' and char != 'N' and char != 't' and char != 'T'):
  char = input("\nOminac trening modelu? (t,n): ")

if (char == 'n' or char == 'N'):
  trainer.fit(modelTSN, daneTS)
  #trainer.fit(modelSN, daneTS)
  torch.save(modelTSN, 'modelTSN.h5')
  print("Trening")
else:
  modelTSN = torch.load('modelTSN.h5')

print("\n********* P O D S U M O W A N I E  W Y N I K O W  T R E N I N G U **********\n")
#Podsumowanie wyników treningu
#%load_ext tensorboard
#tensorboard –logdir logs
#tensorboard --logdir logs
#tensorboard --logdir = '/logs/'
#tensorboard --logdir "logs/"
#tensorboard --logdir=runs


   
    
print("\n********* F U N K C J A  W I Z U A L I Z A C J I  T R A N S F O R M A C J I **********\n")
#definicja transformacji

transform = transforms.Compose([
                                    #transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0), (1))
                                ])
                                
 #funkcja wizualizacji transformacji
def visualize_tsd(dataloader, od, do):
    iterable = iter(dataloader)
    for i in range(od):
      input_tensor = next(iterable)[0]
    for i in range(od,do):
      with torch.no_grad():
        input_tensor = next(iterable)[0]
        #print(input_tensor)
        #input_tensor = input_tensor.to(device)
        #input_tensor = data
        transformed_tensor = transform(input_tensor).permute(1,2,0)
        transformed_tensor = transformed_tensor.to(device)
        #input_grid = torchvision.utils.make_grid(input_tensor)
        #input_grid = convert_image_np(input_tensor)
        transformed_grid = torchvision.utils.make_grid(transformed_tensor)
        #transformed_grid = convert_image_np(transformed_tensor)
        transformed_tensor = transformed_tensor.cpu()
        print (i, input_tensor)
        # Plot the results side-by-side
        #plt.figure(num=i, figsize=(6,8))
        fig = plt.figure(num=i+1, figsize=(8,4))
        ax = fig.subplots(1,2)
        #fig, ax = plt.subplots(1, 2)
        #fig.set_size_inches((8,4))
        #fig.num = i
        ax[0].imshow(input_tensor)
        ax[0].set_title('Dataset Images')
        #ax[0].axis('off')

        ax[1].imshow(transformed_tensor)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')
        plt.show()


visualize_tsd(train_dataset, 0, 3)


#print("\n********* F U N K C J A  W I Z U A L I Z A C J I  D E T E K C J I  **********\n")

#Rozpakowanie zbiorów pokazowych
#!unzip test-images.zip

#!unzip widoki.zip

#funkcja wizualizacji detekcji przy pomocy cascady
#import PIL

cascade = cv2.CascadeClassifier("cascade.xml")
model = modelTSN
#model = torch.load('modelTSN.h5')
#model.eval()
#file = open("signTnames.csv")
#labelTSnames = file.read().strip().split("\n")[1:]
labelnames = [lab.split(",")[1] for lab in labelTSnames]

def visualize_detect(path, namefile):
    print(namefile)
    img = cv2.imread(path+namefile)
    if (img is None):
       print("Nie znaleziono pliku ", namefile)
    else:
      arr_image = np.array(img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      plt.imshow(img)
      plt.title(namefile)
      plt.show()
      #print(type(img))
      try:
         print(img.shape)
      except AttributeError:
         print("shape not found")
      #detection
      cv2.ocl.setUseOpenCL(False)
      img_out = cv2.imread(path+namefile)
      img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
      boxes = cascade.detectMultiScale(img_out, scaleFactor=1.01, minNeighbors=7, minSize=(24,24), maxSize=(224,224))
      print('Liczba wykrytych znaków: ',len(boxes))
      if (len(boxes) > 0):
         #recognition and drawing boundary boxes on input image
         for (x,y,w,h) in boxes:
           print('\nbox: ',x,y,w,h)
           img_rect = cv2.rectangle(img_out,(x,y),(x+w,y+h),(255,255,0),2)

           cropped_image = arr_image[y:y+h,x:x+w, : ]
           cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
           cropped_image = PIL.Image.fromarray(cropped_image)
           cropped_image = transform(cropped_image).permute(1,2,0)
           plt.figure(num=i,figsize=(4,4))
           plt.title("Wykryty znak")
           plt.imshow(cropped_image)
           plt.show()
           cropped_image = cropped_image.permute(2,0,1)
           cropped_image = np.expand_dims(cropped_image, axis=0)

           cropped_image = torch.from_numpy(cropped_image)

           cropped_image = modelSN(cropped_image)
           preds = F.softmax(cropped_image, dim=1)
           print("predictions: ", preds.detach().numpy())
           label = preds.argmax( axis=1)[0]
           label = label.item()
           value = preds.max()   #.item()
           if (value > 0.01):
             labname = labelnames[label]
             labnameS = labelnames[label].split(" ")[0]
             print(f"Nr znaku: {label}, max pred: {value:.3f}, znak: {labname}")
             cv2.putText(img_out,labnameS,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,0,0),2)
           else:
             print(namefile, f", max pred: {value:.3f} zbyt małe prawdopodobieństwo predykcji")

      plt.figure(num=i, figsize=(6,4))
      plt.imshow(img_out)
      plt.title("Zlokalizowane znaki")
      plt.show()
      img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
      cv2.imwrite('test-images/Output/'+ namefile, img_out)
    
    
print("\n*********   W I Z U A L I Z A C J A  D E T E K C J I  O B R A Z K A  **********\n")
#wizualizacja detekcji pojedynczego obrazka

namefile = 'input00.jpg'
path = ''  # 'test-images/Input/'
i = 0
plt.figure(num=i, figsize=(6,4))
visualize_detect(path, namefile)

print("\n*********  W I Z U A L I Z A C J A  D E T E K C J I  F O L D E R U  I N P U T **********")
if(kontynuacja()):



  #wizualizacja detekcji całego folderu Input
  path = 'test-images/'
  subpath ='Input/'
  #test_images = ImageFolder(root=path)
  print("Liczba obrazów:",len(os.listdir(path+subpath)))
  i = 0
  for image in sorted(os.listdir(path+subpath)):
    i = i+1
    print(f"\n{i}.")
    plt.figure(num=i, figsize=(6,4))
    visualize_detect(path+subpath,image)


print("\n*********  D E M O   Z A P I S A N Y C H   E F E K T O W  D E T E K C J I **********\n")
#przegląd zapisanych efektów detekcji

path = 'test-images/Output/'
#outputs = ImageFolder(root=path)
print("Liczba obrazów:",len(os.listdir(path)))
i = 0
for namefile in sorted(os.listdir(path)):
   i = i+1
   print(f"{i}.  {namefile}")
   image = cv2.imread(path+namefile)
   if (image is None):
     print("Nie znaleziono pliku ", namefile)
   else:
     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     plt.figure(num=i, figsize=(6,4))
     plt.title("Przeglad efektow")
     plt.imshow(img)
     plt.show()

 

print('\n******************** PODSUMOWANIE ********************\n')

print("           ")
print("           ")
print("           ")  
print("           ")


print("\nUwaga: zakończenie programu  po zamknięciu diagramu!")
plt.show()





'''
print("\nWstępna analiza danych  W Y K R E S Y")
#wykresy kolumnowe
plt.figure(num=3, figsize=(12,8), dpi=60)
plt.subplot(2, 2, 1)
cars['typ paliwa'].value_counts().plot(kind='bar',color='green')
plt.title("Diagram typów paliwa")
plt.ylabel('liczba aut')
plt.xlabel('typ paliwa')
#plt.xticks([0,1],['benzyna', 'diesel'])
  
plt.subplot(2, 2, 2)
cars['typ silnika'].value_counts().plot(kind='bar',color='blue')
plt.title("Diagram typów silnika")
plt.ylabel('liczba aut')
plt.xlabel('typ silnika')

plt.subplot(2, 2, 3)  
sns.countplot(cars['typ nadwozia'])
plt.title('Typ nadwozia')
plt.xlabel('liczba aut', fontsize=10)

plt.subplot(2, 2, 4)  
sns.countplot(cars['naped'], color='orange')
plt.title('Typ napędu')
plt.xlabel('liczba aut', fontsize=10)
plt.subplots_adjust(top=0.97, bottom=0.05, left=0.12, right=0.98)
plt.tight_layout(h_pad=0.3)
plt.savefig('Figura3.jpeg', dpi=400)
plt.show()
 
#Wykresy korelacji 
fig = plt.figure(num=4, figsize=(12,9), dpi=60)
fig.suptitle("Punktowo-liniowe wykresy korelacji", fontsize=16)
print("\n--- Wydruk Pojemnosc silnika vs cena ---") 
plt.subplot(2, 2, 1)
# wykres punktowo-liniowy (nachylenie linii wskazuje na korelację między „wielkością silnika” a „ceną”.
sns.regplot(x="pojemnosc silnika", y="cena", data= cars)
plt.title("Pojemnosc silnika vs cena")
plt.ylim(0,)
print(cars[["pojemnosc silnika","cena"]].corr())
print('Nachylenie linii wykresu 1 wskazuje na dodatnią korelację między „pojemnością silnika” a „ceną”')
print("Pojemność silnika jest dobrym wyznacznikiem ceny")
   
print("\n--- Wydruk Moc silnika vs cena ---")
plt.subplot(2, 2, 2)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem mocy silnika rośnie cena)
sns.regplot(x="moc silnika", y="cena", data=cars, color='blue')
plt.title("Moc silnika vs cena")
print(cars[["moc silnika","cena"]].corr())
print('Nachylenie linii wykresu 2 wskazuje na dodatnią korelację między „mocą silnika” a „ceną”')
   
print("\n--- Wydruk Spalanie-miasto vs cena ---")
plt.subplot(2, 2, 3)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem masy własnej auta rośnie cena)
sns.regplot(x="spalanie-miasto", y="cena", data=cars, color = 'red')
plt.title("Spalanie-miasto vs cena")
print(cars[["spalanie-miasto","cena"]].corr())
print('Nachylenie linii wykresu 3 wskazuje na dodatnią korelację między „spalanie-miasto” a „ceną”')
   
print("\n--- Wydruk Długość auta vs cena ---")
plt.subplot(2, 2, 4)
# wykres punktowo-liniowy rozrzutu (wraz ze wzrostem długości auta rośnie cena)
sns.regplot(x="dlugosc", y="cena", data=cars, color = 'green')
plt.title("Długość auta vs cena")
print(cars[["dlugosc","cena"]].corr())
print('Nachylenie linii wykresu 4 wskazuje na dodatnią korelację między „długością auta” a „ceną”')

plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95)
plt.tight_layout(h_pad=0.4)
plt.savefig('Figura4.jpeg', dpi=400)
plt.show()
     
print("\n*************************** PRZETWARZANIE **************************")
# podział cars na X i y
X = cars.loc[:, ['symbol', 'typ paliwa', 'turbo/std', 'liczba drzwi',
       'typ nadwozia', 'naped', 'dlugosc', 'szerokosc', 'wysokosc', 'masa',
       'typ silnika', 'liczba cylindrow', 'pojemnosc silnika', 'moc silnika',
       'spalanie-miasto', 'spalanie-autostrada'
       ]]

y = cars['cena']

# przekształcanie zbioru X
print("\nKolumny z wartościami typu obiekt")
cars_categorical = X.select_dtypes(include=['object'])   #lista kolumn z elementami typu object
print(cars_categorical.head())

#Konwertowanie zmiennych kategorycznych w zmienne manekinowe
print("\nTworzenie zamiennych kolumn tzw. manekinów")
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True, dtype=float) #tworzenie nowych kolumn z wartosciami 0 lub 1 
print(cars_dummies.head())
print("Usuwanie kolumn z wartościami kategorycznymi")
X = X.drop(list(cars_categorical.columns), axis=1)  #usuwanie kolumn z wartościami kategorycznymi
print(X.head())
print("Dołączanie nowo utworzonych kolumn")
X = pd.concat([X, cars_dummies], axis=1)  #dołączanie nowo utworzonych kolumn
print(X.head())

print("\nPodział danych na treningowe i testowe oraz skalowanie danych")
# podział na train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=100)

scaler = Normalizer().fit(X)
scale_features = ['symbol', 'liczba drzwi', 'dlugosc', 'szerokosc', 'wysokosc', 'masa', 
       'pojemnosc silnika', 'moc silnika', 'spalanie-miasto', 'spalanie-autostrada'
       ] 
#print("Kolumny z wartościami liczbowymi")       
#print(scale_features)
print("\nZbiór treningowy X_train")
X_train[scale_features] = scaler.fit_transform(X_train[scale_features])
print(X_train[:10])
print('Rozmiar X_train ', X_train.shape)
print("\nZbiór testowy X_test")
X_test[scale_features] = scaler.fit_transform(X_test[scale_features])
print(X_test.head())
print('Rozmiar X_test ',X_test.shape)
print("\nCeny rzeczywiste zbioru testowego y_test")
print(y_test.head())
print('Rozmiar y_test  ',y_test.shape)

# przetwarzanie
print("\n   W Y N I K I  model Ridge (GridSearchCV)")
print("\nWyszukiwanie przez GridSearchCV najlepszych wartości parametrów dla estymatora Ridge.")
params={'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3,
                                   0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50,
                                   100, 500, 1000]}
model = GridSearchCV(estimator = Ridge(), 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 0)            
model.fit(X_train, y_train)
#cv_results = pd.DataFrame(model_Ridge_cv.cv_results_)
#print(cv_results.head())
print('Najlepszy parametr: ', model.best_params_)
alfa = model.best_params_["alpha"]
#najlepszy model
ridge = Ridge(alpha=alfa)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
df= pd.DataFrame({'cena':y_test,'cena_pred':y_pred})
df['cena']= round(df['cena'], 2)
df['cena_pred']= round(df['cena_pred'], 2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = np.sqrt(mean_squared_error(y_test, y_pred))/wsp    #RMSE(pierwiastek kwadratowy błędu średniokwadratowego)
R2 = r2_score(y_test, y_pred)                               #NRMSE znormalizowany RMSE(podzielony przez maxcena-mincena)
print()                                                       #R-Squared - współczynnik determinacji R2, dopasowania modelu do danych uczących 
# Ocena modelu
print('Ocena modelu ', Ridge(alpha=alfa))
print('NRMSE:',NRMSE, '    R-Squared:',R2)

plt.figure(num=5, figsize=(14,8), dpi=60)
plt.subplot(2, 3, 1)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test,y_pred)
plt.title('Cena vs Prognoza - model Ridge')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
plt.ylabel('prognozowana cena', fontsize=10)
plt.ylim(0, 180000)

indeks= [i for i in range(1, y_pred.size+1, 1)]
plt.subplot(2, 3, 4)
plt.plot(indeks,y_test, color = 'red', linewidth=2, linestyle="-")
plt.plot(indeks,y_pred, linewidth=2, linestyle="-")
plt.title('Wykresy cen')
plt.xlabel('indeks', fontsize=10)
plt.ylabel('cena', fontsize=10)
plt.legend(["cena","cena_pred"], loc=1, borderpad=0.15)
plt.ylim(0, 200000)

print("\n   W Y N I K I  model LinearRegression")
#from sklearn.cross_validation import train_test_split

linearmodel = LinearRegression()
linearmodel.fit(X_train,y_train)
score = linearmodel.score(X_test, y_test)   #R-Squared
y_pred = linearmodel.predict(X_test)
print(y_pred[:61])
df = pd.DataFrame()
df['cena'] = y_test
df['cena']= round(df['cena'],2)
df['cena-pred'] = y_pred
df['cena-pred']= round(df['cena-pred'],2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = np.sqrt(mean_squared_error(y_test, y_pred))/wsp
print()
# Ocena modelu
print('Ocena modelu ',LinearRegression())
print('NRMSE:',NRMSE, '    R-Squared:', score)

plt.subplot(2, 3, 2)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test,y_pred, color='green')
plt.title('Cena vs Prognoza - model LinearRegression')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
#plt.ylabel('prognoza ceny', fontsize=0)
plt.ylim(0, 180000)

indeks= [i for i in range(1, y_pred.size+1, 1)]
plt.subplot(2, 3, 5)
plt.plot(indeks,y_test, color = 'red', linewidth=2, linestyle="-")
plt.plot(indeks,y_pred, color = 'green', linewidth=2, linestyle="-")
plt.title('Wykresy cen')
plt.xlabel('indeks', fontsize=10)
plt.legend(['cena','cena_pred'], loc=1, borderpad=0.15)
plt.ylim(0, 200000)

print("\n   W Y N I K I  model svm.SVR")
clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)   #R-Squared
y_pred = clf.predict(X_test)

df = pd.DataFrame()
df['cena'] = y_test
df['cena']= round(df['cena'],2)
df['cena-pred'] = y_pred
df['cena-pred']= round(df['cena-pred'],2)
df['Nazwa auta'] = cars['Nazwa auta']
print(df.head(10))
maxcena = cars['cena'].max()
mincena = cars['cena'].min()
wsp = maxcena - mincena
NRMSE = np.sqrt(mean_squared_error(y_test, y_pred))/wsp
print()
# Ocena modelu
print('Ocena modelu ',svm.SVR(kernel='linear'))
print('NRMSE:',NRMSE, '    R-Squared:', score)

plt.subplot(2, 3, 3)
sns.regplot(x = y_test, y = y_pred, color='red')
plt.scatter(y_test,y_pred, color='blue')
plt.title('Cena vs Prognoza - model svm.SVR')
plt.xlabel('cena', fontsize=10)
plt.xticks([0, 40000, 80000, 120000, 160000, 200000])
plt.ylim(0, 180000)

indeks= [i for i in range(1, y_pred.size+1, 1)]
plt.subplot(2, 3, 6)
plt.plot(indeks,y_test, color = 'red', linewidth=2, linestyle="-")
plt.plot(indeks,y_pred, color = 'blue', linewidth=2, linestyle="-")
plt.title('Wykresy cen', fontsize = 12)
plt.xlabel('indeks', fontsize=10)
plt.ylim(0, 200000)
plt.subplots_adjust(top=0.96, bottom=0.06, left=0.07, right=0.99)
plt.tight_layout(h_pad=0.4)
plt.legend(["cena","cena_pred"], loc=1, borderpad=0.15)
plt.savefig('Figura5.jpeg', dpi=400)
'''


