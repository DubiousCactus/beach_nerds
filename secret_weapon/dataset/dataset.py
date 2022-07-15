# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:44
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

from collections import namedtuple
import os
import json
import pickle
from PIL import Image
import torch
import numpy as np

from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.misc import convert_path
from utils.image import imread
from utils.box.bbox_np import xy42xywha


class DetDataset(Dataset):
    def __init__(self, root, image_set, names, aug=None, color_space="RGB"):
        self.names = names
        self.aug = aug
        self.color_space = color_space
        self.label2name = dict((label, name) for label, name in enumerate(self.names))
        self.name2label = dict((name, label) for label, name in enumerate(self.names))
        self._images, self._labels = self.load_dataset(root, image_set)

    def load_dataset(self, root, image_set):
        print("[*] Loading dataset...")
        images, labels = [], []
        if image_set == "trainval":
            json_file = os.path.join(root, "challenge", "train_challenge.json")
            imgs_path = os.path.join(root, "processed", "train")
            box_labels_file = os.path.join(root, "challenge", "train_box_labels.np")
        elif image_set == "test":
            json_file = os.path.join(root, "json", "challenge", "test_challenge.json")
            imgs_path = os.path.join(root, "raw")
            box_labels_file = None
        else:
            raise Exception(f"No dataset for {image_set}")

        with open(json_file, "r") as f:
            json_file = json.load(f)
        if image_set == "trainval" and box_labels_file is not None:
            # Load bbox labels
            with open(box_labels_file, "rb") as f:
                bbox_labels = pickle.load(f)
            img_list = json_file.keys()
            for img in tqdm(img_list):
                img_path = os.path.join(imgs_path, img)
                images.append(
                    # (np.asarray(Image.open(os.path.join(imgs_path, img)).convert("RGB")), img)
                    (img_path, imread(img_path, self.color_space))
                )
                img_labels = [
                    {
                        "rotated_rect": img_label[0],
                        "bbox": img_label[1],
                        "nr_bbox": None,
                        "area": None,
                        "name": "barcode",
                    }
                    for img_label in bbox_labels[img]
                ]
                labels.append(img_labels)
        else:
            raise NotImplementedError()
            img_list = json_file.keys()
            for img in tqdm(img_list):
                img_path = os.path.join(imgs_path, img)
                images.append(
                    # (np.asarray(Image.open(os.path.join(imgs_path, img)).convert("RGB")), img)
                    (img_path, imread(img_path, self.color_space))
                )
                # masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                img_labels = [
                    {
                        "boxes": img_label["boxes"],
                        "area": (img_label[2][3] - img_label[2][1])
                                * (img_label[2][2] - img_label[2][0]),
                    }
                    for img_label in json_file[img]
                ]
                labels.append(img_labels)
        return images, labels

    @staticmethod
    def load_objs(gt_list, name2label=None):
        nr_bboxes = [
            gt["nr_bbox"] for gt in gt_list
        ]  # "Original" bbox (horizontally aligned)
        bboxes = [gt["bbox"] for gt in gt_list]  # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4
        areas = [gt["area"] for gt in gt_list]
        labels = [
            name2label[gt["name"]] if name2label else gt["name"] for gt in gt_list
        ]
        # Assume all instances are not crowd
        n_objs = len(bboxes)
        iscrowd = torch.zeros((n_objs,), dtype=torch.int64)
        objs = {
            "non_r_bboxes": np.array(nr_bboxes, dtype=np.float32),
            "bboxes": np.array(bboxes, dtype=np.float32),
            "areas": np.array(areas, dtype=np.float32),
            "iscrowd": iscrowd,
            "labels": np.array(labels),
        }
        return objs

    @staticmethod
    def convert_objs(objs):
        target = dict()
        if objs:
            # Limit the angle between -45° and 45° by set flag=2
            target["bboxes"] = torch.from_numpy(
                np.stack([xy42xywha(bbox, flag=2) for bbox in objs["bboxes"]])
            ).float()
            target["labels"] = torch.from_numpy(objs["labels"]).long()
        return target

    def __getitem__(self, index):
        (img_path, img), gt = self._images[index], self._labels[index]
        objs = self.load_objs(gt, self.name2label)
        info = {
            "img_path": img_path,
            "image_id": torch.tensor([index]),
            "shape": img.shape,
            "objs": objs,
        }
        if self.aug is not None:
            img, objs = self.aug(img, deepcopy(objs))
        # import matplotlib.pyplot as plt
        # import cv2
        # for obj in objs['bboxes']:
        # obj = np.int0(obj)
        # print(obj)
        # print(len(obj))
        # cv2.drawContours(img, [obj], 0, (0,255,0))
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        return img, objs, info

    @staticmethod
    def collate(batch):
        images, targets, infos = [], [], []
        # Ensure data balance when parallelizing
        batch = sorted(batch, key=lambda x: len(x[1]["labels"]) if x[1] else 0)
        for i, (img, objs, info) in enumerate(batch):
            images.append(torch.from_numpy(img).reshape(*img.shape[:2], -1).float())
            targets.append(DetDataset.convert_objs(objs))
            infos.append(info)
        return torch.stack(images).permute(0, 3, 1, 2), targets, infos

    def __len__(self):
        return len(self._images)
