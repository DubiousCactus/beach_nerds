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
import torch
import argparse
import numpy as np

# PyTorch Imports
from torch.utils.data import DataLoader

from dataset.aug import ops
from dataset import DetDataset
from dataset.aug.compose import Compose
from dataset.data_utilities import LoggiPackageDataset, get_transform, collate_fn

from model.rdd import RDD
from model.backbone import resnet

from utils.box.rbbox_np import rbbox_batched_nms
from utils.box.bbox_np import xy42xywha, xywha2xy4

from utils.coco_eval import CocoEvaluator, convert_to_coco_api


DATA_DIR="data"
PREDICTIONS_DIR="predictions"

# Function: Compute VISUM 2022 Competition Metric
def visum2022score(bboxes_mAP, masks_mAP, bboxes_mAP_weight=0.5):

    # Compute masks_mAP_weight from bboxes_mAP_weight
    masks_mAP_weight = 1 - bboxes_mAP_weight

    # Compute score, i.e., score = 0.5*bboxes_mAP + 0.5*masks_mAP
    score = (bboxes_mAP_weight * bboxes_mAP) + (masks_mAP_weight * masks_mAP)

    return score

def evaluate(model, val_loader, img_size):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    coco = convert_to_coco_api(val_loader.dataset)
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    model.eval()
    ret_raw = defaultdict(list)
    nms_thresh = 0.45
    for images, targets in tqdm.tqdm(val_loader):
        # Preprocessing
        # TODO: cpu!!
        images = torch.stack([img.cuda() for img in images], dim=0)
        dets = model(images)
        for det, targets in zip(dets, targets):
            if det:
                bboxes, scores, labels = det
                bboxes = bboxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                fname = (targets["image_id"].item(), targets["image_fname"])

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
        outputs = {"boxes": [], "labels": [], "scores": [], "masks": []}
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = xywha2xy4(bbox).ravel()
            line = "%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (
                fname[1],
                score,
                *bbox,
            )
            # print(line)
            ret_save[label].append(line)
            # TODO: compute bounding box of rotated rect (easy with opencv)
            int_rect = np.int0(bbox)
            corners = np.array([[int_rect[0], int_rect[1]], [int_rect[2], int_rect[3]],
                [int_rect[4], int_rect[5]], [int_rect[6], int_rect[7]]])
            xmin, ymin, w, h = cv2.boundingRect(corners) # x, y, w, h
            # convert to xmin, ymin, xmax, ymax
            # ACTUALLY, coco uses the same format as opencv!
            # bbox = np.array([xmin, ymin, xmin+w, ymin+h])
            # compute segmentation mask from rotated rect
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            cv2.fillPoly(mask, [corners], color=255)
            mask = np.expand_dims(mask, axis=0)
            mask =np.swapaxes(mask, 0, 2)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(mask)
            # plt.show()
            # bundle together in COCO format, as such:
            outputs['boxes'].append(np.array([xmin, ymin, w, h]))
            outputs['labels'].append(label)
            outputs['masks'].append(mask)
            outputs['scores'].append(score)
        outputs = {k: torch.tensor(np.array(v)) for k, v in outputs.items()}
        res = {fname[0]: outputs}
        coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def main(args):
    image_size = args.img_size
    batch_size = 1
    num_workers = 8

    #  ============= Build dataset =========================
    test_transforms = get_transform(data_augment=False, img_size=image_size)
    test_set = LoggiPackageDataset(
        data_dir=DATA_DIR, training=False, transforms=test_transforms
    )
    # TODO: Remove
    import torch
    indices = torch.randperm(len(test_set)).tolist()
    test_set = torch.utils.data.Subset(test_set, indices[:800])

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    num_classes = 1

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
    model.restore("tsar.pt")
    model.cuda()

    eval_results = evaluate(model, test_loader, image_size)

    # Get the bounding-boxes results (for VISUM2022 Score)
    bbox_results = eval_results.coco_eval["bbox"]
    bbox_map = bbox_results.stats[0]

    # Get the segmentation results (for VISUM2022 Score)
    segm_results = eval_results.coco_eval["segm"]
    segm_map = segm_results.stats[0]

    # Compute the VISUM2022 Score
    visum_score = visum2022score(bbox_map, segm_map)

    # Print mAP values
    print(f"Detection mAP: {np.round(bbox_map, 4)}")
    print(f"Segmentation mAP: {np.round(segm_map, 4)}")
    print(f"VISUM Score: {np.round(visum_score, 4)}")


    # Save visum_score into a metric.txt file in the PREDICTIONS_DIR
    with open(os.path.join(PREDICTIONS_DIR, "metric.txt"), "w") as m:
        m.write(f"{visum_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backbone", default="resnet50", type=str, choices=["resnet50", "resnet101"]
    )
    parser.add_argument("--img_size", default=512, type=int)

    args = parser.parse_args()
    main(args)
