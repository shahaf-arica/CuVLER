from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import argparse


def read_coco_ann_as_results(coco_ann_file):
    """
    This function reads a coco annotation file and converts it to a results file with a given score
    :param coco_ann_file: path to coco annotation file
    :param score: score to give the annotations
    :return: results
    """
    coco = COCO(coco_ann_file)

    results = []

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "score" in ann:
                score = ann["score"]
            else:
                score = 1.0
            if "segmentation" in ann:
                segmentation = ann["segmentation"]
            else:
                segmentation = []
            results.append({
                "image_id": img_id,
                "category_id": ann["category_id"],
                "score": score,
                "bbox": [round(b) for b in ann["bbox"]],
                "segmentation": segmentation
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script make coco evaluation on a given coco gt annotation file and a results file or pseudo labels annotation file')
    parser.add_argument('--gt_ann_file', type=str, default="datasets/imagenet/annotations/imagenet_val_cls_agnostic.json",
                        help='path of coco annotation file')
    parser.add_argument('--res_file', type=str, default="votecut_annotations_imagenet_val.json",
                        help='path of results file  to evaluate.')
    parser.add_argument('--pseudo_labels', action="store_true", help='If the results file is a pseudo labels file (coco annotation file)')
    parser.add_argument('--iou_type', type=str, default="bbox", choices=["bbox", "segm"])

    args = parser.parse_args()

    coco_gt = COCO(args.gt_ann_file)
    if args.pseudo_labels:
        res = read_coco_ann_as_results(args.res_file)
    else:
        res = args.res_file


    coco_res = coco_gt.loadRes(res)
    coco_eval = COCOeval(coco_gt, coco_res, iouType=args.iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
