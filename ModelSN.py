import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from torch import optim

modelSN = models.squeezenet1_1(pretrained=True)


class ModelSN(pl.LightningModule):
    def __init__(self, num_classes=92):
        super().__init__()

        self.backbone = modelSN.features
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 92)

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
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pooling(x).flatten(1)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
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

        # target = outputs.round().long()
        # tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='multilabel', threshold=0.5)

        # self.train_roc_curve = metrics.roc_auc_score(tp,fp)
        # self.train_roc_curve(outputs, labels)
        # self.log('train_roc_curve', self.train_roc_curve, on_epoch=True, on_step= False)

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

        # target = outputs.round().long()
        # tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='multilabel', threshold=0.5)

        # self.val_roc_curve = metrics.roc_auc_score(tp,fp)
        # self.train_roc_curve(outputs, labels)
        # self.log('train_roc_curve', self.train_roc_curve, on_epoch=True, on_step= False)

        return loss
