# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py
# Code Modified by Shahaf Arica from https://github.com/facebookresearch/CutLER/blob/main/data/datasets/builtin.py

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .builtin_meta import _get_builtin_metadata
from .coco import register_coco_instances
from .meta_coco import register_meta_coco

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO_CA = {}
_PREDEFINED_SPLITS_COCO_CA["coco_cls_agnostic"] = {
    # coco class-agnostic annotations for evaluation
    "cls_agnostic_coco_val_17": ("coco/val2017", "/home/ssaricha/CutLER/datasets/coco/annotations/coco_cls_agnostic_instances_val2017.json"),
    # coco 20K class-agnostic annotations for evaluation
    "cls_agnostic_coco20k": ("coco/train2014", "/strg/C/shared-data/coco/annotations/coco20k_trainval_gt.json"),
    # self-training coco train 17 annotations from detector output
    "coco_train_17_self_training": ("coco/train2017", "coco/annotations/coco_cls_agnostic_instances_train2017_thresh0.2.json"),
}


_PREDEFINED_SPLITS_IMAGENET = {}
_PREDEFINED_SPLITS_IMAGENET["imagenet"] = {
    # VoteCut iou_clustering annotations
    "votecut_imagenet_train": ("imagenet/annotations", "imagenet/annotations/imagenet_votecut_pl_Kmax_3_tuam_0.2.json"),
    # imagenet class agnostic annotations for evaluation - has only bbox annotations
    "cls_agnostic_imagenet_val": ("imagenet/annotations", "imagenet/annotations/imagenet_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_VOC = {}
_PREDEFINED_SPLITS_VOC["voc"] = {
    'cls_agnostic_voc': ("voc/", "voc/annotations/trainvaltest_voc_2007_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_CROSSDOMAIN = {}
_PREDEFINED_SPLITS_CROSSDOMAIN["cross_domain"] = {
    'cls_agnostic_clipart': ("clipart/", "clipart/annotations/traintest_clipart_cls_agnostic.json"),
    'cls_agnostic_watercolor': ("watercolor/", "watercolor/annotations/traintest_watercolor_cls_agnostic.json"),
    'cls_agnostic_comic': ("comic/", "comic/annotations/traintest_comic_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_LVIS = {}
_PREDEFINED_SPLITS_LVIS["lvis"] = {
    "cls_agnostic_lvis": ("coco/", "coco/annotations/lvis1.0_cocofied_val_cls_agnostic.json"),
}


_PREDEFINED_SPLITS_OpenImages = {}
_PREDEFINED_SPLITS_OpenImages["openimages"] = {
    'cls_agnostic_openimages': ("openImages/validation", "openImages/annotations/openimages_val_cls_agnostic.json"),
}


def register_all_imagenet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_IMAGENET.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_voc(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_cross_domain(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CROSSDOMAIN.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_all_openimages(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OpenImages.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_coco_ca(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_CA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )




_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_all_coco_ca(_root)
register_all_imagenet(_root)
register_all_voc(_root)
register_all_cross_domain(_root)
register_all_openimages(_root)
register_all_lvis(_root)
