
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import random
import PIL

import matplotlib.pyplot as plt

dataset = ImageFolder(root='C:\\NeuroBrave\\Scrapy\\kaggle_ml\\data')

dataset_size = len(dataset)
dataset_size

# Data augmentation
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.RandomCrop(128, padding=8, padding_mode='reflect'),
    # T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(*imagenet_stats, inplace=True),
    # T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
device

test_size = 200
nontest_size = len(dataset) - test_size

nontest_df, test_df = random_split(dataset, [nontest_size, test_size])
len(nontest_df), len(test_df)

val_size = 200
train_size = len(nontest_df) - val_size

train_df, val_df = random_split(nontest_df, [train_size, val_size])
len(train_df), len(val_df)

test_df.dataset.transform = valid_tfms
val_df.dataset.transform = valid_tfms

train_df.dataset.transform = train_tfms

batch_size = 64

train_dl = DataLoader(train_df, batch_size, shuffle=True,
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_df, batch_size * 2,
                    num_workers=2, pin_memory=True)
test_dl = DataLoader(test_df, batch_size * 2,
                     num_workers=2, pin_memory=True)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


class CnnModel2(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.wide_resnet101_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 62)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


model = to_device(CnnModel2(), device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


model = to_device(CnnModel2(), device)
model.load_state_dict(torch.load('C:\\NeuroBrave\\Scrapy\\kaggle_ml'))
evaluate(model, val_dl)['val_acc']
evaluate(model, test_dl)['val_acc']

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()
img, label = test_df[0]
plt.imshow(img[0], cmap='gray')
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])
random_image = np.random.randint(0, len(test_df))
print('Random image number ', random_image)
img, label = test_df[random_image]
plt.imshow(img[0], cmap='gray')
print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])