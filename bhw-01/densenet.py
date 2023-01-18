import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import seaborn as sns
from IPython.display import clear_output
from tqdm.notebook import tqdm
import wandb

CLASS_QUANTITY = 200


class TrainValDataset(Dataset):
    def __init__(self, img_dir_path, csv_file_path, train=True, val_size=0.25, transform=None, random_seed=42):
        super().__init__()
        self.img_dir_path = img_dir_path
        self.transform = transform

        self.targets = {}  # targets[название файла картинки] = индекс класса
        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.targets[row['Id']] = int(row['Label'])

        image_files = np.array(os.listdir(self.img_dir_path))
        np.random.seed(random_seed)
        np.random.shuffle(image_files)
        val_size = int(val_size * image_files.shape[0])
        if train:
            self.image_filenames = list(image_files[val_size:])
        else:
            self.image_filenames = list(image_files[:val_size])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir_path, self.image_filenames[item])
        image = Image.open(img_path).convert('RGB')
        image_class = self.targets[self.image_filenames[item]]

        if self.transform:
            image = self.transform(image)

        return image, image_class


class TestDataset(Dataset):
    def __init__(self, img_dir_path, transform=None):
        super().__init__()
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.image_filenames = os.listdir(self.img_dir_path)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir_path, self.image_filenames[item])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_filenames[item]


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', weights=None)
        self.model.classifier = nn.Linear(in_features=1024, out_features=200, bias=True)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=1)


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc, device):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, labels in tqdm(train_loader, desc=tqdm_desc):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


@torch.no_grad()
def validation_epoch(model, criterion, test_loader, tqdm_desc, device):
    test_loss, test_accuracy = 0.0, 0.0
    model.eval()
    for images, labels in tqdm(test_loader, desc=tqdm_desc):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        test_loss += loss.item() * images.shape[0]
        test_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


def full_train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs, device):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device
        )

        test_loss, test_accuracy = validation_epoch(
            model, criterion, test_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}',
            device=device
        )

        if scheduler is not None:
            scheduler.step(test_loss)

        wandb.log({'train_accuracy': train_accuracy,
                   'train_loss': train_loss,
                   'test_accuracy': test_accuracy,
                   'test_loss': test_loss})


def train_classifier(num_epochs=5, lr=1e-3, batch_size=128):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_transform = T.Compose([
        T.Resize(224),
        # T.CenterCrop(224),
        T.RandomRotation((-20, 20)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.3),
        T.ToTensor(),
        normalize
    ])

    val_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        normalize,
    ])

    train_dataset = TrainValDataset(img_dir_path='/home/jupyter/mnt/datasets/bhw1/trainval/trainval',
                                    csv_file_path='/home/jupyter/mnt/datasets/bhw1/labels.csv',
                                    train=True,
                                    val_size=0.25,
                                    transform=train_transform)

    val_dataset = TrainValDataset(img_dir_path='/home/jupyter/mnt/datasets/bhw1/trainval/trainval',
                                  csv_file_path='/home/jupyter/mnt/datasets/bhw1/labels.csv',
                                  train=False,
                                  val_size=0.25,
                                  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # logging to WANDB
    run = wandb.init(project="bhw-01-dl", config={
        "learning_rate": lr,
        'resize': 224,
        "epochs": num_epochs,
        "batch_size": batch_size,
        'aug': 'resize rot flip jit',
        "optimizer": 'SGD',
        "scheduler": 'OneCycleLR(three_phase=True, max_lr=1e-1, pct_start=0.4',
        'model': 'densenet161'
    })
    print(wandb.config)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    model = MyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, three_phase=True, max_lr=1e-1, pct_start=0.4, epochs=num_epochs, steps_per_epoch=1)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=5, mode='triangular2')
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)
    """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=1)"""
    criterion = torch.nn.CrossEntropyLoss()
    full_train(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs, device=device)
    run.finish()
    # torch.save(model.state_dict(), "good_model.pth")
    return model


def apply_classifier(model, test_folder):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    test_transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        normalize,
    ])

    test_dataset = TestDataset(img_dir_path=test_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = []
    model.eval()
    for img, img_path in tqdm(test_loader):
        img_class = model.predict(img).detach().numpy().ravel().item()
        results.append({
            'Id': img_path[0],
            'Label': img_class
        })
    return results


def get_test_labels():
    model = MyModel()
    model.load_state_dict(torch.load('good_model.pth'))
    model.eval()
    results = apply_classifier(model, '/home/jupyter/mnt/datasets/bhw1/test/test')
    results = pd.DataFrame(results)
    results.to_csv('labels_test.csv', index=False)


if __name__ == '__main__':
    train_classifier(num_epochs=30, lr=1e-2)
    # get_test_labels()

