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
import tqdm
import torch
import wandb
import argparse
import numpy as np

# PyTorch Imports
from torch import optim
from torch.utils.data import DataLoader

from dataset.aug import ops
from dataset import DetDataset
from dataset.aug.compose import Compose

from model.rdd import RDD
from model.backbone import resnet

from utils.box.rbbox_np import rbbox_batched_nms
from utils.adjust_lr import adjust_lr_multi_step
from utils.box.bbox_np import xy42xywha, xywha2xy4
from utils.parallel import convert_model, CustomDetDataParallel


def evaluate(model, label2name, val_loader, image_size):
    model.eval()
    ret_raw = defaultdict(list)
    nms_thresh = 0.45
    for images, targets, infos in tqdm.tqdm(val_loader):
        # Preprocessing
        images = images.cuda()
        dets = model(images, targets)
        for (det, info) in zip(dets, infos):
            if det:
                bboxes, scores, labels = det
                bboxes = bboxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                fname, x, y, w, h = os.path.splitext(
                    os.path.basename(info["img_path"])
                )[0].split("-")[:5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                long_edge = max(w, h)
                pad_x, pad_y = (long_edge - w) // 2, (long_edge - h) // 2
                bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                bboxes *= long_edge / image_size
                bboxes -= [pad_x, pad_y]
                bboxes += [x, y]
                bboxes = np.stack([xy42xywha(bbox) for bbox in bboxes])
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
            ret_save[label2name[label]].append(line)


def main(args):
    dir_dataset = "../data_participants"
    dir_save = "log"
    dir_weight = os.path.join(dir_save, "weight")
    os.makedirs(dir_weight, exist_ok=True)

    image_size = args.img_size
    batch_size = args.batch_size
    num_workers = 4

    config = dict(
        learning_rate=0.001,
        architecture=args.backbone,
        batch_size=args.batch_size,
    )

    wandb.init(
        project="VISUM",
        notes="Visum 2022",
        config=config,
        entity="beach_nerds",
    )

    #  ============= Build dataset =========================
    train_aug = Compose(
        [
            ops.ToFloat(),  # Standardizes to [0,1]
            # ops.PhotometricDistort(),  # TODO: Move to grayscale and remove
            ops.RandomHFlip(),
            ops.RandomVFlip(),
            ops.RandomRotate90(),
            # ops.ResizeJitter([0.8, 1.2]),
            # ops.PadSquare(),
            # ops.Normalize(
            #     [51.61898139, 51.61898139, 51.61898139],
            #     [50.11639468, 50.11639468, 50.11639468],
            # ),  # Our dataset [0, 255]
            ops.Normalize(
                [0.20242738, 0.20242738, 0.20242738],
                [0.19653489, 0.19653489, 0.19653489],
            ),  # Our dataset [0, 1]
            # ops.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
            ops.Resize(image_size),
        ]
    )
    val_aug = Compose(
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
    dataset = DetDataset(
        dir_dataset, "trainval", ["barcode"], aug=train_aug, color_space="RGB"
    )
    dataset_notransforms = DetDataset(
        dir_dataset, "trainval", ["barcode"], aug=val_aug, color_space="RGB"
    )

    # Split the dataset into train and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    train_pct, n_samples = 0.8, len(indices)
    n_train = int(train_pct * n_samples)
    n_val = n_samples - n_train
    print(f"[*] Using {n_train} training samples and {n_val} validation samples")
    # Train Set: 1100 samples
    train_set = torch.utils.data.Subset(dataset, indices[:n_train])
    # Validation Set: 199 samples
    val_set = torch.utils.data.Subset(dataset_notransforms, indices[n_train:])

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset_notransforms.collate,
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
    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model.init()
    if torch.cuda.is_available():
        print(f"[*] Using cuda")
        model = model.cuda()

    # ===================== Train ====================================

    if args.opt == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, 1e-9, verbose=True
        )
    elif args.scheduler == "expo":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, verbose=True)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5, verbose=True
        )

    print(f"[*] Using scheduler {args.scheduler}")
    print(f"[*] Using optimiser {args.opt}")

    alpha = 1.0
    best_loss = float("+inf")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}/{args.epochs}")
        print("Training Phase")
        model.train()

        batches, train_loss, train_loc_loss, train_cls_loss = 0, 0.0, 0.0, 0.0
        for images, targets, infos in tqdm.tqdm(train_loader):
            # Preprocessing
            images = images.cuda()

            optimizer.zero_grad()
            losses = model(images, targets)
            loss = losses["loss_cls"] + (alpha * losses["loss_loc"])
            train_loss += loss.detach().item()
            train_loc_loss += losses["loss_loc"].detach().item()
            train_cls_loss += losses["loss_cls"].detach().item()
            loss.backward()
            optimizer.step()
            batches += 1

        train_loss /= batches
        train_loc_loss /= batches
        train_cls_loss /= batches
        wandb.log(
            {
                "train/loc_loss": train_loc_loss,
                "train/cls_loss": train_cls_loss,
                "train/total_loss": train_loss,
            },
            step=epoch,
        )
        print(
            f"[*] Training loss={train_loss} (loc={train_loc_loss}, cls={train_cls_loss})"
        )

        # Validation
        if ((epoch + 1) % args.val_every == 0) or (epoch == args.epochs - 1):
            print("Validation Phase")
            with torch.no_grad():
                batches, val_loss, val_loc_loss, val_cls_loss = 0, 0.0, 0.0, 0.0
                for images, targets, infos in tqdm.tqdm(val_loader):
                    # Preprocessing
                    images = images.cuda()
                    losses = model(images, targets)
                    val_loss += losses["loss_cls"] + (alpha * losses["loss_loc"])
                    val_loc_loss += losses["loss_loc"].detach().item()
                    val_cls_loss += losses["loss_cls"].detach().item()
                    batches += 1
                val_loss /= batches
                val_cls_loss /= batches
                val_loc_loss /= batches
                wandb.log(
                    {
                        f"val/total_loss": val_loss,
                        "val/loc_loss": val_loc_loss,
                        "val/cls_loss": val_cls_loss,
                    },
                    step=epoch,
                )
                print(
                    f"[*] Validation loss={val_loss} (loc={val_loc_loss}, cls={val_cls_loss})"
                )
                if val_loss < best_loss:
                    print("[*] New best loss!")
                    best_loss = val_loss
                    file_name = f"visum2022_loss-{val_loss:06f}.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                        },
                        os.path.join(dir_weight, file_name),
                    )
                    print(
                        f"Model successfully saved at {os.path.join(dir_weight, file_name)}"
                    )
        if args.scheduler is not None:
            scheduler.step()

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backbone", default="resnet50", type=str, choices=["resnet50", "resnet101"]
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument(
        "--opt", default="radam", type=str, choices=["adamw", "adam", "radam"]
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument(
        "--scheduler", default="cosine", choices=[None, "cosine", "expo", "step"]
    )
    parser.add_argument(
        "--val_every", default=1, type=int, help="Validate every n epochs"
    )
    parser.add_argument("--img_size", default=512, type=int)

    args = parser.parse_args()
    main(args)
