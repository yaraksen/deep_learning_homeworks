import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnext50_32x4d
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import os
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import seaborn as sns
from IPython.display import clear_output
from tqdm.notebook import tqdm
import wandb

CLASS_QUANTITY = 200

os.environ["WANDB_API_KEY"] = '###'


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


def white_list_wd(model, lr):
    with_decay, without_decay = set(), set()
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters():
            name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name.endswith("bias"):
                without_decay.add(name)
            elif param_name.endswith("weight"):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    with_decay.add(name)
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    without_decay.add(name)

    model_parameters = {param_name: param for param_name, param in model.named_parameters()}
    regularize_params = [model_parameters[param] for param in sorted(list(with_decay))]
    no_regularize_params = [model_parameters[param] for param in sorted(list(without_decay))]
    optimizer = torch.optim.SGD([
        {"params": regularize_params, "weight_decay": 2e-05},
        {"params": no_regularize_params, "weight_decay": 0.0},
    ], lr=lr, momentum=0.9)
    return optimizer


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnext50_32x4d(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.BatchNorm2d(2048),
            nn.Linear(in_features=2048, out_features=CLASS_QUANTITY, bias=True),
        )

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
    last_acc = 0.0
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
            scheduler.step()

        if test_accuracy > last_acc:
            torch.save(model.state_dict(), f"model_acc{test_accuracy}[{datetime.now()}].pth")
            print('saved better model with acc =', test_accuracy)
            last_acc = test_accuracy

        print(f'lr={scheduler.get_last_lr()[0]}')
        wandb.log({'train_accuracy': train_accuracy,
                   'train_loss': train_loss,
                   'test_accuracy': test_accuracy,
                   'test_loss': test_loss,
                   'lr': scheduler.get_last_lr()[0]})


def train_classifier(num_epochs, lr, batch_size, checkpoint=None):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.RandomErasing(p=0.1),
        normalize
    ])

    val_transform = T.Compose([
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    train_dataset = TrainValDataset(img_dir_path='../input/bhw-1-deep-learning/bhw1-dataset/trainval/',
                                    csv_file_path='../input/bhw-1-deep-learning/bhw1-dataset/labels.csv',
                                    train=True,
                                    val_size=0.25,
                                    transform=train_transform)

    val_dataset = TrainValDataset(img_dir_path='../input/bhw-1-deep-learning/bhw1-dataset/trainval/',
                                  csv_file_path='../input/bhw-1-deep-learning/bhw1-dataset/labels.csv',
                                  train=False,
                                  val_size=0.25,
                                  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    model = MyModel().to(device)
    if checkpoint is not None:
        print('start from checkpoint:', checkpoint)
        model.load_state_dict(torch.load(checkpoint))

    lr = 0.5
    optimizer = white_list_wd(model, lr=lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    lr_warmup_epochs = 5
    lr_warmup_decay = 0.01

    main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - lr_warmup_epochs, eta_min=0)
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs)

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs], verbose=True
    )

    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    run_config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        'aug': train_transform,
        "optimizer": optimizer,
        "scheduler": 'SequentialLR',
        'model': model
    }

    # logging to WANDB
    run = wandb.init(project="bhw-01-dl", config=run_config)
    full_train(model, optimizer, lr_scheduler, criterion, train_loader, val_loader, num_epochs, device=device)
    # get_test_labels(model, run)
    run.finish()
    torch.save(model.state_dict(), f"model[{datetime.now()}].pth")
    return model


def apply_classifier(model, test_folder):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = T.Compose([
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    test_dataset = TestDataset(img_dir_path=test_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = []
    model.eval()
    for img, img_path in tqdm(test_loader):
        img_class = model.predict(img.to(device)).cpu().detach().numpy().ravel().item()
        results.append({
            'Id': img_path[0],
            'Label': img_class
        })
    return results


@torch.no_grad()
def get_test_labels(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    results = apply_classifier(model, '../input/bhw-1-deep-learning/bhw1-dataset/test/')
    results = pd.DataFrame(results)
    results.to_csv(f'labels_test[{datetime.now()}].csv', index=False)


if __name__ == '__main__':
    train_classifier(300, 0.5, 128)
