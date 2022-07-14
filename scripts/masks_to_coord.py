#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Convert segmentation masks to quadrilateral coordinates.
"""


import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import pickle
import cv2

from PIL import Image
from time import sleep


def convert_mask_to_coords(mask):
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(mask)
    # Returns: center, (width, height), rotation_angle
    rect = cv2.minAreaRect(points)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    box = np.int0(cv2.boxPoints(rect))
    # print(rect, box)
    # cv2.drawContours(mask, [box], 0, (0,255,0))
#     plt.figure()
#     plt.imshow(mask)
#     plt.show()
    return rect, box


def main(labels_path, masks_path, output_path):
    labels = {}
    # Load the labels file
    with open(labels_path, "r") as labels_json:
        labels = json.load(labels_json)

    # For each input, go through all masks
    for input_img, ground_truth in labels.items():
        labels[input_img] = []
        # Load the mask image
        for mask_file in ground_truth['masks']:
            mask_path = os.path.join(masks_path, "train", input_img.split(".")[0], mask_file)
            # Convert to coordinates
            mask_img = np.asarray(Image.open(mask_path).convert("L"))
            r_rect, coords = convert_mask_to_coords(mask_img)
            # Save in dict
            labels[input_img].append((r_rect, coords))

    # Export final dict as numpy file
    with open(output_path, "wb") as f:
        pickle.dump(labels, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("labels_path", type=str)
    argparser.add_argument("masks_path", type=str)
    argparser.add_argument("output_path", type=str)
    args = argparser.parse_args()
    main(args.labels_path, args.masks_path, args.output_path)
