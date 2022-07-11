#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from PIL import Image

import shutil
import os


_root = "data_participants/"
size = 512
crop_sz = 512

for folder in ["masks", "processed"]:
    for root, _, files in os.walk(os.path.join(_root, folder)):
        for file in files:
            if not file.endswith("jpg"):
                continue
            img_path = os.path.join(root, file)
            while img_path.endswith('.old'):
                shutil.move(img_path, f"{img_path[:-4]}")
                img_path = f"{img_path[:-4]}"
            moved_path = os.path.join(root, file + ".old")
            shutil.move(img_path, moved_path)
            img = Image.open(moved_path)
            w, h = img.size
            if w == size:
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
