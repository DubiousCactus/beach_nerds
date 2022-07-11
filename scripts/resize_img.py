#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from PIL import Image
from tqdm import tqdm

import shutil
import os


_root = "data/"
size = 512
crop_sz = 512

_subjects = [
        "20200709-subject-01",
        "20200813-subject-02",
        "20200820-subject-03",
        "20200903-subject-04",
        "20200908-subject-05",
        "20200918-subject-06",
        "20200928-subject-07",
        "20201002-subject-08",
        "20201015-subject-09",
        "20201022-subject-10",
    ]

_viewpoints = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]
pbar = tqdm(total=len(_subjects)*len(_viewpoints)*100)
for n in _subjects:
    seq = [
            os.path.join(n, s)
            for s in sorted(os.listdir(os.path.join(_root, n)))
            ]
    for i, q in enumerate(seq):
        # Fetch samples and compute labels for each camera
        for c in _viewpoints:
            for root, _, files in os.walk(os.path.join(_root, q, c)):
                for file in files:
                    if file.startswith("aligned_depth"):
                        os.remove(os.path.join(root, file))
                        continue
                    if not file.startswith("color"):
                        continue
                    img_path = os.path.join(root, file)
                    while img_path.endswith('.old'):
                        shutil.move(img_path, f"{img_path[:-4]}")
                        img_path = f"{img_path[:-4]}"
                    moved_path = os.path.join(root, file + ".old")
                    shutil.move(img_path, moved_path)
                    img = Image.open(moved_path)
                    w, h = img.size
                    if w == 224:
                        continue
                    new_size = size, size
                    if w > h:
                        new_size = (size, h*size//w)
                    new_img = img.resize(new_size)
                    width, height = new_img.size   # Get dimensions
                    left = (width - crop_sz)//2
                    top = (height - crop_sz)//2
                    right = (width + crop_sz)//2
                    bottom = (height + crop_sz)//2

                    # Crop the center of the image
                    new_img = new_img.crop((left, top, right, bottom))
                    new_img.save(img_path)
                    img.close()
                    new_img.close()
                    os.remove(moved_path)
            pbar.update()
