import os
import json
from pycocotools import mask as mask_util
import datetime
from tqdm import tqdm
from pathlib import Path
from pycocotools import mask
import shutil
import numpy as np
from glob import glob

INFO = {
    "description": "ImageNet train-set: VoteCut pseudo-masks",
    "url": "",
    "version": "1.0",
    "year": 2024,
    "contributor": "Shahaf Arica",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "CC-BY 4.0 License",
        "url": "https://creativecommons.org/licenses/by/4.0/deed.en"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

category_info = {
    "is_crowd": 0,
    "id": 1
}


def get_info(contributor="", desc="", url="", year="", version="",):
    info = INFO.copy()
    if contributor:
        info["contributor"] = contributor
    if desc:
        info["description"] = desc
    if url:
        info["url"] = url
    if year:
        info["year"] = year
    if version:
        info["version"] = version
    return info


def get_license_info(name="", url=""):
    license = LICENSES.copy()
    if name:
        license[0]["name"] = name
    if url:
        license[0]["url"] = url
    return license


def collect_to_single_ann_dict(anns_files, imgnet_train=False,
                               contributor="", desc="", url="", year="", version="", license_url=""):
    """
    Collects all the temp annotations files into a single coco annotation dict
    :param anns_files: list of temp annotation files
    :param imgnet_train: if True, the images are from the imagenet train set arranged in subfolders by category
    :param contributor: contributor name for the info field
    :param desc: description for the info field
    :param url: url for the info field
    :param year: year for the info field
    :param version: version for the info field
    :param license_url: license url for the license field
    :return:
    """
    info = get_info(contributor, desc, url, year, version)
    license = get_license_info("Apache License", license_url)
    categories = CATEGORIES.copy()

    images_info = []
    ann_info = []
    ann_id = 1
    images = set()
    all_images = set()  # for debug

    for file in tqdm(anns_files, desc="Creating coco annotations dict"):
        with open(file, "r") as f:
            job_out_data = json.load(f)
        for image_data in job_out_data:
            img_file_name = Path(image_data["file_name"]).name
            if imgnet_train:
                category_name = img_file_name.split("_")[0]
                file_name = f"{category_name}/{img_file_name}"
            else:
                file_name = img_file_name
            any_crf_success = False
            max_cluster_size = 0
            for ann in image_data["annotations"]:
                if ann["crf_success"]:
                    any_crf_success = True
                max_cluster_size = max(max_cluster_size, ann["cluster_size"])

            if img_file_name not in images:
                all_images.add(image_data["image_id"])  # for debug
                image_dict = {
                    "id": image_data["image_id"],
                    "file_name": file_name,
                    "height": image_data["height"],
                    "width": image_data["width"],
                    datetime.datetime.utcnow().isoformat(' '): datetime.datetime.utcnow().isoformat(' '),
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "max_cluster_size": max_cluster_size,
                    "any_crf_success": any_crf_success
                }
                images_info.append(image_dict)
                images.add(img_file_name)

                for ann in image_data["annotations"]:
                    ann_dict = {
                        "id": ann_id,
                        "image_id": image_data["image_id"],
                        "category_id": 1,
                        "segmentation": ann["segmentation"],
                        "area": mask.area(ann["segmentation"]).tolist(),
                        "bbox": ann["bbox"],
                        "iscrowd": 0,
                        "height": image_data["height"],
                        "width": image_data["width"],
                        "crf_success": ann["crf_success"],
                        "cluster_size": ann["cluster_size"],
                        "score": ann["cluster_size"] / max_cluster_size
                    }
                    ann_id += 1
                    ann_info.append(ann_dict)

    print("Number of images: {}".format(len(images_info)))
    print("Number of annotations: {}".format(len(ann_info)))
    coco_ann_dict = {
        "info": info,
        "licenses": license,
        "categories": categories,
        "images": images_info,
        "annotations": ann_info,
    }
    return coco_ann_dict


def create_ann_for_single_image(image_id, file_name, height, width, image_masks):
    annotations = []
    for i, mask_data in enumerate(image_masks):
        rle = mask_util.encode(np.asfortranarray(mask_data["mask"]))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({
            "bbox": mask_util.toBbox(rle).tolist(),
            "segmentation": rle,
            "crf_success": mask_data["crf_success"],
            "cluster_size": mask_data["cluster_size"]
        })
    if  len(annotations) > 0:
        return {
            "file_name": file_name,
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": annotations
        }
    else:
        return None


class CocoAnnotationsWorker:
    """
    A worker class for handling temp coco annotations files
    """
    def __init__(self, worker_dir):
        self.worker_dir = worker_dir
        self.ann_dicts = []
        self.num_files = 0

    def add_image_ann(self, image_id, file_name, height, width, image_masks):
        image_ann = create_ann_for_single_image(image_id, file_name, height, width, image_masks)
        self.num_files += 1
        if image_ann is not None:
            self.ann_dicts.append(image_ann)
            return True
        else:
            return False

    def flush_and_save_anns(self):
        """
        Flushes the current annotations to a temp file
        """
        with open(os.path.join(self.worker_dir, f"ann_{self.num_files-1}.json"), "w") as f:
            json.dump(self.ann_dicts, f)
        self.ann_dicts.clear()

    def __len__(self):
        return len(self.ann_dicts)

    def cleanup(self):
        self.cleanup_tmp_files(self.worker_dir)

    @classmethod
    def collect_to_single_ann_dict(cls, anns_files, imgnet_train=False,
                                   contributor="", desc="", url="", year="", version="", license_url=""):
        return collect_to_single_ann_dict(anns_files, imgnet_train, contributor, desc, url, year, version, license_url)

    @classmethod
    def cleanup_tmp_files(cls, worker_dir):
        shutil.rmtree(worker_dir, ignore_errors=True)
        
    def resume(self, all_img_files):
        all_worker_ann_files = glob(os.path.join(self.worker_dir, "ann_*.json"))
        anns = []
        for ann_file in all_worker_ann_files:
            with open(ann_file, "r") as f:
                anns.extend(json.load(f))
        existing_files = [a["file_name"] for a in anns]
        self.num_files = len(anns)
        img_files = set(all_img_files) - set(existing_files)
        return img_files

    def done(self):
        # save a flag file to indicate the worker is done
        Path(os.path.join(self.worker_dir, "done")).touch()

    def is_done(self):
        return os.path.exists(os.path.join(self.worker_dir, "done"))

