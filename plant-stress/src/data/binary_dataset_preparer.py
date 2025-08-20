#!/usr/bin/env python3
import os
import re
import random
import shutil
from pathlib import Path

"""
Prepare a binary health dataset (healthy vs diseased) from PlantDoc folders by symlinking.
Usage:
  python scripts/prepare_health_binary_dataset.py \
    --src "/Users/sohamkolhe/Downloads/PlantDoc-Dataset" \
    --dst "data_proc/health_binary" --val-ratio 0.15
"""

import argparse


DISEASE_TERMS = [
    'rust', 'blight', 'spot', 'mildew', 'mold', 'virus', 'rot', 'septoria', 'bacterial', 'late', 'early'
]


def is_healthy(class_name: str) -> bool:
    n = class_name.lower()
    # Healthy classes in PlantDoc typically end with 'leaf' and lack disease terms
    if not n.endswith('leaf'):
        return False
    return not any(term in n for term in DISEASE_TERMS)


def symlink_images(src_dirs, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for d in src_dirs:
        for img in d.glob('*.jpg'):
            link_path = dst_dir / img.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(img)
                except FileExistsError:
                    pass
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='PlantDoc-Dataset root (has train/test)')
    p.add_argument('--dst', default='data_proc/health_binary', help='Output dataset root')
    p.add_argument('--val-ratio', type=float, default=0.15)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    train_root = src / 'train'
    test_root = src / 'test'

    # List class dirs
    train_classes = [d for d in train_root.iterdir() if d.is_dir()]

    healthy_dirs = [d for d in train_classes if is_healthy(d.name)]
    diseased_dirs = [d for d in train_classes if not is_healthy(d.name)]

    # Prepare train/val split from train
    train_h_dir = dst / 'train' / 'healthy'
    train_d_dir = dst / 'train' / 'diseased'
    val_h_dir = dst / 'val' / 'healthy'
    val_d_dir = dst / 'val' / 'diseased'

    for d in [train_h_dir, train_d_dir, val_h_dir, val_d_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Symlink with simple random split per class
    random.seed(42)
    for group, out_train, out_val in [
        (healthy_dirs, train_h_dir, val_h_dir),
        (diseased_dirs, train_d_dir, val_d_dir),
    ]:
        for cls_dir in group:
            imgs = sorted(cls_dir.glob('*.jpg'))
            if not imgs:
                continue
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs) * args.val_ratio))
            val_imgs = imgs[:n_val]
            trn_imgs = imgs[n_val:]
            for img in trn_imgs:
                link = out_train / f"{cls_dir.name.replace(' ', '_')}_{img.name}"
                if not link.exists():
                    try:
                        link.symlink_to(img)
                    except FileExistsError:
                        pass
            for img in val_imgs:
                link = out_val / f"{cls_dir.name.replace(' ', '_')}_{img.name}"
                if not link.exists():
                    try:
                        link.symlink_to(img)
                    except FileExistsError:
                        pass

    # Build test set from PlantDoc test
    test_classes = [d for d in test_root.iterdir() if d.is_dir()]
    test_h_dir = dst / 'test' / 'healthy'
    test_d_dir = dst / 'test' / 'diseased'
    for d in [test_h_dir, test_d_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for cls_dir in test_classes:
        out = test_h_dir if is_healthy(cls_dir.name) else test_d_dir
        for img in cls_dir.glob('*.jpg'):
            link = out / f"{cls_dir.name.replace(' ', '_')}_{img.name}"
            if not link.exists():
                try:
                    link.symlink_to(img)
                except FileExistsError:
                    pass

    print('Done. Dataset at:', dst)


if __name__ == '__main__':
    main()


