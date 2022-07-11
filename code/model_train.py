#!/bin/env python

# Imports
import os
import tqdm
import argparse
import numpy as np

# PyTorch Imports
import torch
import torch.utils.data

# Tensorboard: PyTorch
from torch.utils.tensorboard import SummaryWriter

# Project Imports
from data_utilities import get_transform, collate_fn, LoggiPackageDataset
from model_utilities import LoggiBarcodeDetectionModel, evaluate, visum2022score


def main(args):
    # Random seeds
    torch.manual_seed(42)
    # Constant variables
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.img_size
    VAL_MAP_FREQ = args.val_every
    last_score = .0

    # Fix the random crashes due to multiprocessing
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # Directories
    DATA_DIR = "data_participants"
    SAVE_MODEL_DIR = "results/models"
    if not os.path.isdir(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)

    # Tensorboard Writer
    tb = SummaryWriter(log_dir="results/tensorboard", flush_secs=10)

    # Prepare data
    # First, we create two train sets with different transformations (we will use the one w/out transforms as validation set)
    dataset = LoggiPackageDataset(
        data_dir=DATA_DIR,
        training=True,
        transforms=get_transform(data_augment=True, img_size=IMG_SIZE),
    )
    dataset_notransforms = LoggiPackageDataset(
        data_dir=DATA_DIR,
        training=True,
        transforms=get_transform(data_augment=False, img_size=IMG_SIZE),
    )

    # Split the dataset into train and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    # Train Set: 1000 samples
    train_set = torch.utils.data.Subset(dataset, indices[:-299])
    # Validation Set: 299 samples
    val_set = torch.utils.data.Subset(dataset_notransforms, indices[-299:])

    # DataLoaders
    # Train loader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Validation loader
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Define DEVICE (GPU or CPU)
    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {DEVICE}")

    # Define model
    model = LoggiBarcodeDetectionModel(
        min_img_size=IMG_SIZE, max_img_size=IMG_SIZE, backbone=args.backbone
    )

    # Print model summary
    model_summary = model.summary()

    # Put model into DEVICE
    model.to(DEVICE)

    # Define an optimizer
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_params, lr=0.001)
    # optimizer = torch.optim.SGD(model_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Start the training and validation loops
    for epoch in range(NUM_EPOCHS):

        # Epoch
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")

        # Training Phase
        print("Training Phase")
        model.train()

        # Initialize lists of losses for tracking
        losses_classifier = list()
        losses_box_reg = list()
        losses_mask = list()
        losses_objectness = list()
        losses_rpn_box_reg = list()
        losses_ = list()

        # Go through train loader
        for images, targets in tqdm.tqdm(train_loader):
            # TODO: Check this training loop and grad computation
            optimizer.zero_grad()

            # Load data
            images = [image.to(DEVICE) for image in images]
            targets_ = [
                {k: v.to(DEVICE) for k, v in t.items() if k != "image_fname"}
                for t in targets
            ]

            # Compute loss
            loss_dict = model(images, targets_)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            # Save loss values
            losses_classifier.append(loss_dict["loss_classifier"].item())
            losses_box_reg.append(loss_dict["loss_box_reg"].item())
            losses_mask.append(loss_dict["loss_mask"].item())
            losses_objectness.append(loss_dict["loss_objectness"].item())
            losses_rpn_box_reg.append(loss_dict["loss_rpn_box_reg"].item())
            losses_.append(loss_value)

            # Optimise models parameters
            losses.backward()
            optimizer.step()

        # Print loss values
        loss = np.sum(losses_) / len(train_set)
        print(f"Loss Classifier: {np.sum(losses_classifier) / len(train_set)}")
        print(f"Loss Box Regression: {np.sum(losses_box_reg) / len(train_set)}")
        print(f"Loss Mask: {np.sum(losses_mask) / len(train_set)}")
        print(f"Loss Objectness: {np.sum(losses_objectness) / len(train_set)}")
        print(f"Loss RPN Box Regression: {np.sum(losses_rpn_box_reg) / len(train_set)}")
        print(f"Loss: {loss}")

        # Print loss values in tensorboard
        tb.add_scalar("loss/clf", np.sum(losses_classifier) / len(train_set), epoch)
        tb.add_scalar("loss/boxreg", np.sum(losses_box_reg) / len(train_set), epoch)
        tb.add_scalar("loss/mask", np.sum(losses_mask) / len(train_set), epoch)
        tb.add_scalar("loss/obj", np.sum(losses_objectness) / len(train_set), epoch)
        tb.add_scalar("loss/rpn", np.sum(losses_rpn_box_reg) / len(train_set), epoch)
        tb.add_scalar("loss/total_loss", np.sum(losses_) / len(train_set), epoch)

        if ((epoch + 1) % VAL_MAP_FREQ == 0) or (epoch == NUM_EPOCHS - 1):
            # Validation Phase
            eval_results = evaluate(model, val_loader, DEVICE)
            bbox_results = eval_results.coco_eval["bbox"]
            segm_results = eval_results.coco_eval["segm"]
            bbox_map = bbox_results.stats[0]
            segm_map = segm_results.stats[0]
            visum_score = visum2022score(bbox_map, segm_map)

            # Print mAP values
            print(f"Detection mAP: {np.round(bbox_map, 4)}")
            print(f"Segmentation mAP: {np.round(segm_map, 4)}")
            print(f"VISUM Score: {np.round(visum_score, 4)}")

            # Print mAP values in tensorboard
            tb.add_scalar("eval/bbox_map", bbox_map, epoch)
            tb.add_scalar("eval/segm_map", segm_map, epoch)
            tb.add_scalar("eval/visum_score", visum_score, epoch)

            if visum_score > last_score:
                file_name = f"visum2022_score-{visum_score:06f}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(SAVE_MODEL_DIR, file_name),
                )
                last_score = visum_score
                print("[*] New best loss!")
                print(
                    f"Model successfully saved at {os.path.join(SAVE_MODEL_DIR, file_name)}"
                )

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", default="mobilenet_v2", type=str, help="Either mobilenet_v2 or resnet50")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument(
        "--val_every", default=1, type=int, help="Validate every n epochs"
    )

    args = parser.parse_args()
    main(args)
