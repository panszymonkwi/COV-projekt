import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
from torch.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear, MaxPool2d


class ModelTSNet(pl.LightningModule):
    def __init__(self, num_classes=92):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )  # [3, 224, 224]
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )  # [64, 224, 224]
        self.bnorm1 = BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )  # [64, 112, 112]
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )  # [128, 112, 112]
        self.bnorm2 = BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )  # [128, 56, 56]
        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )  # [256, 56, 56]
        self.bnorm3 = BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )  # [256,28, 28]
        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )  # [512, 28, 28]
        self.bnorm4 = BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(512 * 14 * 14, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1024, 92)

        self.loss_function = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes
        )

        self.train_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )
        self.val_macro_f1 = torchmetrics.F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        )

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
            x = x.reshape(x.shape[0], -1)
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
    # tensorboard_initialized = False

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.forward(inputs.float())

        loss = self.loss_function(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        outputs = F.softmax(outputs, dim=1)

        self.train_accuracy(outputs, labels)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, on_step=False)

        self.train_precision(outputs, labels)
        self.log("train_precision", self.train_precision, on_epoch=True, on_step=False)

        self.train_recall(outputs, labels)
        self.log("train_recall", self.train_recall, on_epoch=True, on_step=False)

        self.train_macro_f1(outputs, labels)
        self.log("train_macro_f1", self.train_macro_f1, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.forward(inputs.float())
        loss = self.loss_function(outputs, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        outputs = F.softmax(outputs, dim=1)

        self.val_accuracy(outputs, labels)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, on_step=False)

        self.val_precision(outputs, labels)
        self.log("val_precision", self.val_precision, on_epoch=True, on_step=False)

        self.val_recall(outputs, labels)
        self.log("val_recall", self.val_recall, on_epoch=True, on_step=False)

        self.val_macro_f1(outputs, labels)
        self.log("val_macro_f1", self.val_macro_f1, on_epoch=True, on_step=False)
        return loss
