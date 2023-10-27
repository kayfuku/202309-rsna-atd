# import packages
import os
import sys
import multiprocessing
from pathlib import Path
import random
from collections import defaultdict
from glob import glob
import pickle
from joblib import Parallel, delayed
import gc
from tqdm.notebook import tqdm
from tabulate import tabulate
import yaml
import datetime
from logging import getLogger
import wandb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, AdamW
from torchvision import models
import torchvision.transforms.v2 as t
from torchvision.transforms.v2 import (Resize, Compose, RandomHorizontalFlip,
                                       ColorJitter, RandomAffine, RandomErasing, ToTensor)
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import timm
import albumentations as A


class AbdominalKLSData(Dataset):

    def __init__(self, cfg, train_df, train_img_dirs, slice_start, stride=10, apply_aug=True, num_fold=5):
        super().__init__()
        self.cfg = cfg
        self.slice_start = slice_start
        self.train_df = train_df
        self.train_img_paths = self._fetch_and_sample_train_images(train_img_dirs, stride=stride)
        self.augmentation = apply_aug

        self.gkf = GroupKFold(n_splits=num_fold)
        groups = np.array([os.path.basename(img_path).split('_')[0] for img_path in self.train_img_paths])
        self.fold_data = list(self.gkf.split(self.train_img_paths, groups=groups))

        self.normalize = Compose([
            # Resize((256, 256), antialias=True),
            # RandomHorizontalFlip(),  # Randomly flip images left-right
            # ColorJitter(brightness=0.2),  # Randomly adjust brightness
            # ColorJitter(contrast=0.2),  # Randomly adjust contrast
            # RandomAffine(degrees=0, shear=10),  # Apply shear transformation
            # RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Apply zoom transformation
            # RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Coarse dropout
            ToTensor(),
        ])

        # augmentation
        # flip
        self.aug_h_flip = A.HorizontalFlip(p=0.5)
        self.aug_v_flip = A.VerticalFlip(p=0.5)
        # elastic and grid
        self.aug_distortion = A.GridDistortion(p=0.5)
        self.aug_elastic = A.ElasticTransform(p=0.5)
        # affine
        self.aug_affine = A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.0, 0.2),
            rotate=(-45, 45),
            shear=(-15, 15),
            p=0.5)
        # self.aug_affine = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8)
        # clahe
        self.aug_clahe = A.CLAHE(p=0.5)
        # bright
        self.aug_bright = A.OneOf([
            A.RandomGamma(gamma_limit=(50, 150), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5)
        ], p=0.5)
        # cutout
        self.aug_cutout = A.CoarseDropout(max_height=8, max_width=8, p=0.5)
        # randomcrop
        self.aug_randomcrop = A.RandomResizedCrop(
            height=256,
            width=256,
            scale=(0.8, 1.0),
            ratio=(3/4, 4/3),
            p=0.5)

    def __len__(self):
        return len(self.train_img_paths)

    def __getitem__(self, idx):
        sample_img_path = self.train_img_paths[idx]
        patient_id = int(os.path.basename(sample_img_path).split('_')[0])

        # preprocess image
        img = self._process_img(sample_img_path)
        # img.shape: (256, 256)

        # augmentation
        if self.augmentation:
            img = self.aug_h_flip(image=img)["image"]
            img = self.aug_v_flip(image=img)["image"]
            img = self.aug_distortion(image=img)["image"]
            img = self.aug_clahe(image=img)["image"]
            img = self.aug_affine(image=img)["image"]
            img = self.aug_bright(image=img)["image"]
            img = self.aug_cutout(image=img)["image"]
            img = self.aug_randomcrop(image=img)["image"]

        img = img.astype('float32') / 255
        # img.shape: (256, 256)

        img = torch.tensor(img, dtype=torch.float).unsqueeze(dim=0)
        # img.shape: (1, 256, 256)
        if self.cfg['model']['model_name'] == 'maxvit_tiny_tf_384.in1k':
            img = Compose([Resize((384, 384), antialias=True)])(img)
        img = self.normalize(img)
        # img.shape: (1, 256, 256)

        # labels
        label = self.train_df[self.train_df.patient_id == patient_id].values[0][1:-1]
        kidney = np.argmax(label[4:7], keepdims=False)
        liver = np.argmax(label[7:10], keepdims=False)
        spleen = np.argmax(label[10:], keepdims=False)

        return {
            'patient_id': patient_id,
            'image': img,
            'kidney': kidney,
            'liver': liver,
            'spleen': spleen,
        }

    def _fetch_and_sample_train_images(self, img_dirs, stride):
        """
        Fetches and samples at least one training image per series from the training directories.
        """
        print('Fetching and sampling training images...')
        paths = []
        patients_to_series_to_img_paths = defaultdict(lambda: defaultdict(list))
        for img_dir in img_dirs:
            for filename in tqdm(os.listdir(img_dir)):
                patient_id, series_id, _ = filename.split('_')
                patients_to_series_to_img_paths[patient_id][series_id].append(os.path.join(img_dir, filename))

            for patient_id, series_to_img_paths in patients_to_series_to_img_paths.items():
                for series_id, imgs in series_to_img_paths.items():
                    # sort by instance number
                    sorted_img_paths = sorted(imgs, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    start_index = int(len(sorted_img_paths) * self.slice_start)
                    end_index = int(len(sorted_img_paths) * (self.slice_start + 0.2))
                    roi = sorted_img_paths[start_index:end_index]
                    for img_path in roi[::stride]:
                        paths.append(img_path)

        return paths

    def _process_img(self, img_path):
        image = cv2.imread(img_path)
        # image = image.astype('float32') / 255
        image = (image.astype('float32') * 255).astype('uint8')
        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyscale = cv2.resize(greyscale, (256, 256))
        return greyscale

    def get_one_fold(self, fold=0):
        train_indices, val_indices = self.fold_data[fold]
        train_data = Subset(self, train_indices)
        val_data = Subset(self, val_indices)
        return train_data, val_data


class AbdominalBEData(Dataset):

    def __init__(self, cfg, train_df, data_path, b_e_pos, sample_b_e_neg, apply_aug=True, num_fold=5):
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.b_e_data = pd.concat([b_e_pos, sample_b_e_neg])
        self.train_img_paths = self._fetch_train_image_paths(data_path, self.b_e_data)
        self.augmentation = apply_aug

        self.gkf = GroupKFold(n_splits=num_fold)
        groups = np.array([os.path.basename(img_path).split('_')[0] for img_path in self.train_img_paths])
        self.fold_data = list(self.gkf.split(self.train_img_paths, groups=groups))

        self.normalize = Compose([
            # Resize((256, 256), antialias=True),
            # RandomHorizontalFlip(),  # Randomly flip images left-right
            # ColorJitter(brightness=0.2),  # Randomly adjust brightness
            # ColorJitter(contrast=0.2),  # Randomly adjust contrast
            # RandomAffine(degrees=0, shear=10),  # Apply shear transformation
            # RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Apply zoom transformation
            # RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Coarse dropout
            ToTensor(),
        ])

        # augmentation
        # flip
        self.aug_h_flip = A.HorizontalFlip(p=0.5)
        self.aug_v_flip = A.VerticalFlip(p=0.5)
        # elastic and grid
        self.aug_distortion = A.GridDistortion(p=0.5)
        self.aug_elastic = A.ElasticTransform(p=0.5)
        # affine
        self.aug_affine = A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.0, 0.1),
            rotate=(-35, 35),
            shear=(-15, 15),
            p=0.5)
        # self.aug_affine = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8)
        # clahe
        self.aug_clahe = A.CLAHE(p=0.5)
        # bright
        self.aug_bright = A.OneOf([
            A.RandomGamma(gamma_limit=(60, 140), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5)
        ], p=0.5)
        # cutout
        self.aug_cutout = A.CoarseDropout(max_height=8, max_width=8, p=0.5)
        # randomcrop
        self.aug_randomcrop = A.RandomResizedCrop(
            height=256,
            width=256,
            scale=(0.8, 1.0),
            ratio=(3/4, 4/3),
            p=0.5)

    def __len__(self):
        return len(self.train_img_paths)

    def __getitem__(self, idx):
        sample_img_path = self.train_img_paths[idx]
        patient_id = int(os.path.basename(sample_img_path).split('_')[0])

        # preprocess image
        img = self._process_img(sample_img_path)
        # img.shape: (256, 256)

        # augmentation
        if self.augmentation:
            img = self.aug_h_flip(image=img)["image"]
            img = self.aug_v_flip(image=img)["image"]
            img = self.aug_distortion(image=img)["image"]
            img = self.aug_clahe(image=img)["image"]
            img = self.aug_affine(image=img)["image"]
            img = self.aug_bright(image=img)["image"]
            img = self.aug_cutout(image=img)["image"]
            img = self.aug_randomcrop(image=img)["image"]

        img = img.astype('float32') / 255
        # img.shape: (256, 256)

        img = torch.tensor(img, dtype=torch.float).unsqueeze(dim=0)
        # img.shape: (1, 256, 256)
        if self.cfg['model']['model_name'] == 'maxvit_tiny_tf_384.in1k':
            img = Compose([Resize((384, 384), antialias=True)])(img)
        img = self.normalize(img)
        # img.shape: (1, 256, 256)

        # labels
        bowel = self.b_e_data[self.b_e_data.filename == sample_img_path].bowel.values[0]
        extravasation = self.b_e_data[self.b_e_data.filename == sample_img_path].extravasation.values[0]

        return {
            'patient_id': patient_id,
            'image': img,
            'bowel': bowel,
            'extravasation': extravasation,
        }

    def _fetch_train_image_paths(self, data_path, b_e_data):
        # get paths from both b_e_pos and sample_b_e_neg
        paths = []
        # add base_path to filename
        b_e_data['filename'] = b_e_data['filename'].apply(lambda x: os.path.join(data_path, x))
        paths.extend(b_e_data['filename'])
        # shuffle
        random.shuffle(paths)
        return paths

    def _process_img(self, img_path):
        image = cv2.imread(img_path)
        # image = image.astype('float32') / 255
        image = (image.astype('float32') * 255).astype('uint8')
        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        greyscale = cv2.resize(greyscale, (256, 256))
        return greyscale

    def get_one_fold(self, fold=0):
        train_indices, val_indices = self.fold_data[fold]
        train_data = Subset(self, train_indices)
        val_data = Subset(self, val_indices)
        return train_data, val_data


# Model Architecure
class KLSNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            model_name=cfg['model']['model_name'],
            pretrained=cfg['model']['pretrained'],
            in_chans=cfg['model']['in_chans'],
            num_classes=cfg['model']['num_classes'],
            global_pool=cfg['model']['global_pool'],
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
        )
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.in_features = self.backbone.num_features  # 1280
        hidden_dim = cfg['model']['hidden_dim']
        self.neck = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg['model']['p_dropout']),
        )

        self.kidney = nn.Linear(hidden_dim, 3)
        self.liver = nn.Linear(hidden_dim, 3)
        self.spleen = nn.Linear(hidden_dim, 3)

        self.cce = nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor(cfg['model']['kls_weights']))

        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.probs = defaultdict(list)
        self.targets = defaultdict(list)
        self.auc_scores = dict()

    def forward(self, x):
        # extract features
        x = self.backbone(x)
        x = self.neck(x)

        # output logits
        kidney = self.kidney(x)
        liver = self.liver(x)
        spleen = self.spleen(x)

        return kidney, liver, spleen

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        kidney = batch['kidney']
        liver = batch['liver']
        spleen = batch['spleen']

        k, l, s = self.forward(inputs)
        k_loss = self.cce(k, kidney)
        l_loss = self.cce(l, liver)
        s_loss = self.cce(s, spleen)
        loss = k_loss + l_loss + s_loss
        self.train_epoch_loss.append(loss.item())

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    # def on_train_epoch_end(self):
    #     avg_loss = np.mean(self.train_epoch_loss)
    #     self.log('avg_train_loss', avg_loss, prog_bar=True)
    #     self.train_epoch_loss.clear()

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        kidney = batch['kidney']
        liver = batch['liver']
        spleen = batch['spleen']

        k, l, s = self.forward(inputs)
        k_loss = self.cce(k, kidney)
        l_loss = self.cce(l, liver)
        s_loss = self.cce(s, spleen)
        loss = k_loss + l_loss + s_loss
        self.val_epoch_loss.append(loss.item())

        self.probs['k'].extend(F.softmax(k, dim=1).detach().cpu().numpy())
        self.probs['l'].extend(F.softmax(l, dim=1).detach().cpu().numpy())
        self.probs['s'].extend(F.softmax(s, dim=1).detach().cpu().numpy())
        self.targets['k'].extend(kidney.detach().cpu().numpy())
        self.targets['l'].extend(liver.detach().cpu().numpy())
        self.targets['s'].extend(spleen.detach().cpu().numpy())

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_epoch_loss)

        for t in ['k', 'l', 's']:
            self.auc_scores[t] = roc_auc_score(
                self.targets.get(t),
                self.probs.get(t),
                multi_class='ovo', labels=[0, 1, 2])

        # self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('val_auc_score_k', self.auc_scores.get('k'), prog_bar=True, sync_dist=True)
        self.log('val_auc_score_l', self.auc_scores.get('l'), prog_bar=True, sync_dist=True)
        self.log('val_auc_score_s', self.auc_scores.get('s'), prog_bar=True, sync_dist=True)
        self.val_epoch_loss.clear()
        self.probs.clear()
        self.targets.clear()
        self.auc_scores.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.cfg['model']['lr']))
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.cfg['model']['lr']))
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass


# Model Architecure
class BENet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            model_name=cfg['model']['model_name'],
            pretrained=cfg['model']['pretrained'],
            in_chans=cfg['model']['in_chans'],
            num_classes=cfg['model']['num_classes'],
            global_pool=cfg['model']['global_pool'],
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
        )
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.in_features = self.backbone.num_features  # 1280
        hidden_dim = cfg['model']['hidden_dim']
        self.neck = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg['model']['p_dropout']),
        )

        self.bowel = nn.Linear(hidden_dim, 2)
        self.extravasation = nn.Linear(hidden_dim, 2)

        self.cce_b = nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor(cfg['model']['b_weights']))
        self.cce_e = nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor(cfg['model']['e_weights']))

        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.probs = defaultdict(list)
        self.targets = defaultdict(list)
        self.auc_scores = dict()

    def forward(self, x):
        # extract features
        x = self.backbone(x)
        x = self.neck(x)

        # output logits
        bowel = self.bowel(x)
        extravsation = self.extravasation(x)

        return bowel, extravsation

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        bowel = batch['bowel']
        extravasation = batch['extravasation']

        b, e = self.forward(inputs)
        b_loss = self.cce_b(b, bowel)
        e_loss = self.cce_e(e, extravasation)
        loss = b_loss + e_loss
        self.train_epoch_loss.append(loss.item())

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    # def on_train_epoch_end(self):
    #     avg_loss = np.mean(self.train_epoch_loss)
    #     self.log('avg_train_loss', avg_loss, prog_bar=True)
    #     self.train_epoch_loss.clear()

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        bowel = batch['bowel']
        extravasation = batch['extravasation']

        b, e = self.forward(inputs)
        b_loss = self.cce_b(b, bowel)
        e_loss = self.cce_e(e, extravasation)
        loss = b_loss + e_loss
        self.val_epoch_loss.append(loss.item())

        self.probs['b'].extend(F.softmax(b, dim=1).detach().cpu().numpy())
        self.probs['e'].extend(F.softmax(e, dim=1).detach().cpu().numpy())
        self.targets['b'].extend(bowel.detach().cpu().numpy())
        self.targets['e'].extend(extravasation.detach().cpu().numpy())

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_epoch_loss)

        for t in ['b', 'e']:
            y_true = np.ravel(self.targets.get(t))
            prob_array = np.array(self.probs.get(t))
            if len(np.unique(y_true)) != 2:
                return -1
            self.auc_scores[t] = roc_auc_score(y_true, prob_array[:, 1])

        # self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('val_auc_score_b', self.auc_scores.get('b'), prog_bar=True, sync_dist=True)
        self.log('val_auc_score_e', self.auc_scores.get('e'), prog_bar=True, sync_dist=True)
        self.val_epoch_loss.clear()
        self.probs.clear()
        self.targets.clear()
        self.auc_scores.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.cfg['model']['lr']))
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=float(self.cfg['model']['lr']))
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
        

