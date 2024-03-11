import os
import gdown
from pathlib import Path
import argparse

_ANN_CLS_AGNOSTIC_GT_GDRIVE_IDS = {
    "imagenet": {"imagenet_val_cls_agnostic_gt.json": "1_4W3cuwo3lW6Ickpi7yYrm-TlX4oIoy8"},
    "coco": {"coco20k_trainval_gt.json": "1Vam0DamGADy_rClswNCIDpheGhiPKmYf",
             "coco_cls_agnostic_instances_val2017.json": "1gABmH0nAHDsZzFaC5X0t9k9FLoBbJdwb",
             "lvis1.0_cocofied_val_cls_agnostic.json": "1FX1RnMTD6meJJnyg9SBParPm-jEKTBHD"},
    "voc" : {"trainvaltest_voc_2007_cls_agnostic.json": "1YaDHOnYCbk6NtszRAQYCnrjcb3d7fiIa"},
    "openImages": {"openimages_val_cls_agnostic.json": "1sX87D2I2aoEYTk0PfSnGS3C1Bmi_rcTy"},
    "comic": {"traintest_comic_cls_agnostic.json": "1zKuDcxvoXK8e704U10Hv2-NJvFnh0-lA"},
    "clipart": {"traintest_clipart_cls_agnostic.json": "16Ra9bMXn-O5UiJ-JyN9mtf77dD1x-4lI"},
    "watercolor": {"traintest_watercolor_cls_agnostic.json": "1m7M7Vizs6uHkhHj-N6XxlgIi9zVg6q30"}
}


_ANN_VOTECUT_GDRIVE_IDS = {
    "imagenet": {"imagenet_val_votecut_kmax_3_tuam_0.2.json": "14fqnSJlbsG58PjKxhHyLYqLjEk0KGmSy",
                 "imagenet_train_votecut_kmax_3_tuam_0.2.json": "10vz02vuZV1ql1QoWmMQzSQrivsIOr7Ke"}
}


_ANN_CUVLER_SELF_TRAIN_GDRIVE_IDS = {
    "coco": {"coco_cls_agnostic_instances_train2017_thresh0.2.json": "13uPXVykyoGXIjC6B00eVHupqMNQdPqjh"}
}


_MODEL_ZOO_GDRIVE_IDS = {
    "cuvler_zero_shot": {"model_cuvler_zero_shot.pth": "16PHrqWvqfgcZfO5IfcpmAxCG2QYaQsEM"},
    "cuvler_self_train": {"model_cuvler_coco_self_train.pth": "1jkAnc5KX45gmwnzcwaHjxTSq5U3-JAYD"}
}

def download(root, download_dict):
    Path(root).mkdir(parents=True, exist_ok=True)
    for file_name, file_id in download_dict.items():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output=os.path.join(root, file_name), quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from gdrive")

    parser.add_argument("--gt-ann", type=str, choices=["all", "imagenet", "coco", "voc", "openImages", "comic", "clipart", "watercolor"], help="Download ground truth annotations")
    parser.add_argument("--votecut-ann", type=str, help="Download VoteCut annotations", choices=["all", "train", "val"])
    parser.add_argument("--self-train-ann", type=str, help="Download Self-train annotations", choices=["coco"])
    parser.add_argument("--model", type=str, help="Model to download", choices=["zero_shot", "self_train"])

    args = parser.parse_args()
    # get datasets_path from DETECTRON2_DATASETS if it is set
    datasets_path = os.environ["DETECTRON2_DATASETS"] if "DETECTRON2_DATASETS" in os.environ else "datasets"

    if args.gt_ann:
        if args.gt_ann == "all":
            for k in _ANN_CLS_AGNOSTIC_GT_GDRIVE_IDS.keys():
                download(os.path.join(datasets_path, k, "annotations"), _ANN_CLS_AGNOSTIC_GT_GDRIVE_IDS[k])
        else:
            download(os.path.join(datasets_path, args.gt_ann, "annotations"), _ANN_CLS_AGNOSTIC_GT_GDRIVE_IDS[args.gt_ann])

    if args.votecut_ann:
        if args.votecut_ann == "all":
            download(os.path.join(datasets_path, "imagenet", "annotations"), _ANN_VOTECUT_GDRIVE_IDS["imagenet"])
        else:
            download(os.path.join(datasets_path, "imagenet", "annotations"), {k: v for k, v in _ANN_VOTECUT_GDRIVE_IDS["imagenet"].items() if args.votecut_ann in k})

    if args.self_train_ann:
        if args.self_train_ann == "coco":
            download(os.path.join(datasets_path, "annotations"), _ANN_CUVLER_SELF_TRAIN_GDRIVE_IDS["coco"])

    if args.model:
        if args.model == "zero_shot":
            download(os.path.join(datasets_path, "models"), _MODEL_ZOO_GDRIVE_IDS["cuvler_zero_shot"])
        elif args.model == "self_train":
            download(os.path.join(datasets_path, "models"), _MODEL_ZOO_GDRIVE_IDS["cuvler_self_train"])

