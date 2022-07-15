#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from collections import defaultdict
import sys

sys.path.append(".")

import os
import cv2
import tqdm
import argparse
import numpy as np

# PyTorch Imports
from torch.utils.data import DataLoader

from dataset.aug import ops
from dataset import DetDataset
from dataset.aug.compose import Compose

from model.rdd import RDD
from model.backbone import resnet

from utils.box.rbbox_np import rbbox_batched_nms
from utils.box.bbox_np import xy42xywha, xywha2xy4


def evaluate(model, label2name, val_loader, image_size):
    model.eval()
    ret_raw = defaultdict(list)
    nms_thresh = 0.45
    for images, targets, infos in tqdm.tqdm(val_loader):
        # Preprocessing
        dets = model(images, targets)
        for (det, info) in zip(dets, infos):
            if det:
                bboxes, scores, labels = det
                bboxes = bboxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                fname = os.path.split(info["img_path"])[-1]

                # TODO: Make sure the predictions match the resolution of the test data!!
                #       I think it should be fine because we are given the test data, so
                #       when we resize it, we resize the labels as well.

                # TODO: Figure out this padding stuff!

                # fname, x, y, w, h = os.path.splitext(
                #     os.path.basename(info["img_path"])
                # )[0].split("-")[:5]
                # x, y, w, h = int(x), int(y), int(w), int(h)
                # long_edge = max(w, h)
                # pad_x, pad_y = (long_edge - w) // 2, (long_edge - h) // 2
                # bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                # bboxes *= long_edge / image_size
                # bboxes -= [pad_x, pad_y]
                # bboxes += [x, y]
                # bboxes = np.stack([xy42xywha(bbox) for bbox in bboxes])
                ret_raw[fname].append([bboxes, scores, labels])

    print("merging results...")
    ret = []

    for fname, dets in ret_raw.items():
        bboxes, scores, labels = zip(*dets)
        bboxes = np.concatenate(list(bboxes))
        scores = np.concatenate(list(scores))
        labels = np.concatenate(list(labels))
        keeps = rbbox_batched_nms(bboxes, scores, labels, nms_thresh)
        ret.append([fname, [bboxes[keeps], scores[keeps], labels[keeps]]])

    print("converting to submission format...")
    ret_raw = defaultdict(list)
    ret_save = defaultdict(list)
    for fname, (bboxes, scores, labels) in ret:
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = xywha2xy4(bbox).ravel()
            line = "%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (
                fname,
                score,
                *bbox,
            )
            print(line)
            ret_save[label2name[label]].append(line)
            # TODO: compute bounding box of rotated rect (easy with opencv)
            print("Rotated rect: ", np.int0(bbox))
            int_rect = np.int0(bbox)
            bbox = cv2.boundingRect(int_rect) # x, y, w, h
            print("Bounding box: ", bbox)
            break
            # TODO: compute segmentation mask from rotated rect (coordinates?)
            # TODO: bundle together in COCO format, as such:
            # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # res = {
            #     target["image_id"].item(): output
            #     for target, output in zip(targets, outputs)
            # }
            # coco_evaluator.update(res)

    return 0, 0




def main(args):
    dir_dataset = "../data_participants"
    image_size = args.img_size
    batch_size = 1
    num_workers = 4

    #  ============= Build dataset =========================
    aug = Compose(
        [
            ops.ToFloat(),
            # ops.PadSquare(),
            # ops.Normalize(
            #     [51.61898139, 51.61898139, 51.61898139],
            #     [50.11639468, 50.11639468, 50.11639468],
            # ),  # Our dataset
            ops.Normalize(
                [0.20242738, 0.20242738, 0.20242738],
                [0.19653489, 0.19653489, 0.19653489],
            ),  # Our dataset [0, 1]
            # ops.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
            ops.Resize(image_size),
        ]
    )
    # TODO: Switch to test data
    dataset = DetDataset(
        dir_dataset, "trainval", ["barcode"], aug=aug, color_space="RGB"
    )
    import torch
    indices = torch.randperm(len(dataset)).tolist()
    test_set = torch.utils.data.Subset(dataset, indices[:5])

    data_loader = DataLoader(
        test_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate,
    )
    num_classes = len(dataset.names)

    # ===================== Build model ==============================
    prior_box = {
        "strides": [8, 16, 32, 64, 128],
        "sizes": [3] * 5,
        "aspects": [[1.5, 3, 5, 8]] * 5,
        "scales": [[2**0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }

    cfg = {
        "prior_box": prior_box,
        "num_classes": num_classes,
        "extra": 2,
    }

    if args.backbone == "resnet50":
        backbone = resnet.resnet50
    elif args.backbone == "resnet101":
        backbone = resnet.resnet101
    else:
        raise NotImplementedError(f"No implementation for backbone {args.backbone}")
    model = RDD(backbone(fetch_feature=True, pretrained=False), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    # TODO: Load state dict!
    model.init()
    map, mseg = evaluate(model, dataset.label2name, data_loader, image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backbone", default="resnet50", type=str, choices=["resnet50", "resnet101"]
    )
    parser.add_argument("--img_size", default=512, type=int)

    args = parser.parse_args()
    main(args)
