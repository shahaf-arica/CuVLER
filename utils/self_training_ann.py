import os
import json
import tqdm
import datetime
import argparse
from pycocotools.coco import COCO


INFO = {
    "description": "COCO train-set 2017: Self-train",
    "url": "",
    "version": "1.0",
    "year": 2024,
    "contributor": "Shahaf Arica",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}


CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]


if __name__ == "__main__":
    # load model arguments
    parser = argparse.ArgumentParser(description='Generate json files for the self-training')
    parser.add_argument('--detectron2-out-dir', type=str,
                        default='coco_train17_outputs',
                        help='Path to model predictions splits dir')
    parser.add_argument('--coco-ann-path', type=str, default='datasets/coco/annotations/instances_train2017.json')
    parser.add_argument('--save-path-prefix', type=str,
                        default='datasets/coco/annotations/coco_cls_agnostic_instances_train2017',
                        help='Path to save the generated annotation file')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Confidence score thresholds')
    args = parser.parse_args()

    self_train_ann_file = f"{args.save_path_prefix}_thresh{args.threshold}.json"

    # get license info from the original coco annotation file
    res_json_file = os.path.join(args.detectron2_out_dir, 'inference', 'coco_instances_results.json')

    with open(res_json_file, "r") as f:
        predictions = json.load(f)

    new_anns = []
    ann_id = 1
    for id, ann in enumerate(tqdm.tqdm(predictions, desc='Filtering low-confidence predictions')):
        if ann['score'] >= args.threshold:
            ann['id'] = ann_id
            ann['area'] = ann['bbox'][-1] * ann['bbox'][-2]
            ann['iscrowd'] = 0
            ann['width'] = ann['segmentation']['size'][0]
            ann['height'] = ann['segmentation']['size'][1]
            new_anns.append(ann)
            ann_id += 1

    ann_coco = COCO(args.coco_ann_path)

    new_dataset = {
        "info": INFO,
        "licenses": ann_coco.dataset['licenses'],
        "categories": CATEGORIES,
        "images": ann_coco.dataset['images'],
        "annotations": new_anns
    }

    # save annotation file
    with open(self_train_ann_file, "w") as f:
        json.dump(new_dataset, f)

    print("Done: {} images; {} anns.".format(len(self_train_ann_file['images']), len(self_train_ann_file['annotations'])))
