






# Prepare Datasets for Pseudo-Labeling, Training and Evaluation




CuVLER handles datasets as in [Detectron2](https://detectron2.readthedocs.io/tutorials/datasets.html).
You should set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets` relative to your current working directory.

## Ground-truth, VoteCut and Self-train annotations <a id="downloader"></a>

For easy download in Linux machines, you can use the following commands to download ground-truth and VoteCut annotations:
```
cd path/to/CuVLER
# for ground-truth annotations
python utils/gdrive_downloader.py --gt-ann {all, imagenet, coco, voc, openImages, comic, clipart, watercolor}
# for votecut imagenet annotaions
python utils/gdrive_downloader.py --votecut-ann {all, train, val}
# for votecut self-train coco annotaions
python utils/gdrive_downloader.py --self-train-ann coco
``` 

## ImageNet
Download ImageNet-1K (ILSVRC) from [here](https://image-net.org/download.php). Create soft links to the imagenet {train, validation} 
directory under `$DETECTRON2_DATASETS/imagenet`.
Also, create a directory for annotations files:
```
cd $DETECTRON2_DATASETS
mkdir imagenet
cd imagenet
ln -s /path/to/ILSVRC/Data/CLS-LOC/train train
ln -s /path/to/ILSVRC/Data/CLS-LOC/val val
mkdir annotations
```

The structure should look like this:
```
imagenet/
  train/
    n01440764/*.JPEG
    n01443537/*.JPEG
    ...
  val/
    *.JPEG
    *.JPEG
    ...
  annotations/
    imagenet_val_cls_agnostic_gt.json
    imagenet_train_votecut_kmax_3_tuam_0.2.json  # generated by VoteCut
```
It best to download the COCO-style class-agnostic annotations with our [downloader](#downloader) script,
script. You also can download them directly here: [Ground-truth val](https://drive.google.com/uc?export=download&id=1_4W3cuwo3lW6Ickpi7yYrm-TlX4oIoy8),
[VoteCut val](https://drive.google.com/uc?export=download&id=14fqnSJlbsG58PjKxhHyLYqLjEk0KGmSy),
[VoteCut train](https://drive.google.com/uc?export=download&id=10vz02vuZV1ql1QoWmMQzSQrivsIOr7Ke).

## COCO, COCO20K, LVIS:
Download COCO 2014 and 2017 from [here](https://cocodataset.org/#download). The expected
structure is:
```
coco/
  annotations/
    instances_{train,val}2017.json
    coco20k_trainval_gt.json
    coco_cls_agnostic_instances_val2017.json
    lvis1.0_cocofied_val_cls_agnostic.json
  {train,val}2017/
    000000000139.jpg
    000000000285.jpg
    ...
  train2014/
    COCO_train2014_000000581921.jpg
    COCO_train2014_000000581909.jpg
    ...
```
Download the COCO-style class-agnostic annotations with our [downloader](#downloader) script,
or you can download directly from here: [coco val](https://drive.google.com/uc?export=download&id=1gABmH0nAHDsZzFaC5X0t9k9FLoBbJdwb),
[coco20k](https://drive.google.com/uc?export=download&id=1Vam0DamGADy_rClswNCIDpheGhiPKmYf), 
[lvis](https://drive.google.com/uc?export=download&id=1FX1RnMTD6meJJnyg9SBParPm-jEKTBHD),
[coco self-train annotations](https://drive.google.com/uc?export=download&id=13uPXVykyoGXIjC6B00eVHupqMNQdPqjh).



## VOC2007:
Download dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit). The expected
structure is:
```
voc/
  annotations/
    trainvaltest_2007_cls_agnostic.json
  VOC2007/
    JPEGImages/
      000001.jpg
      ...
```
COCO-style annotation can be downloaded with our [downloader](#downloader) script, 
or directly from [here](https://drive.google.com/uc?export=download&id=1YaDHOnYCbk6NtszRAQYCnrjcb3d7fiIa).


## OpenImages-V6:
Download the OpenImages validation set [here](https://storage.googleapis.com/openimages/web/download_v6.html).
The expected structure is:
```
openImages/
  annotations/
    openimages_val_cls_agnostic.json
  validation/
    47947b97662dc962.jpg
    ...
```
COCO-style annotation can be downloaded with our [downloader](#downloader) script, 
or directly from [here](https://drive.google.com/uc?export=download&id=1sX87D2I2aoEYTk0PfSnGS3C1Bmi_rcTy).

## Comic, Clipart, Watercolor:
For download follow the instruction in [cross-domain-detection](https://github.com/naoto0804/cross-domain-detection).
The expected structure is:

```
clipart/
  annotations/
    traintest_clipart_cls_agnostic.json
  JPEGImages/
    375390294.jpg
    ...
comic/
  annotations/
    traintest_comic_cls_agnostic.json
  JPEGImages/
    161067391.jpg
    ...
watercolor/
  annotations/
    traintest_watercolor_cls_agnostic.json
  JPEGImages/
    163330523.jpg
    ...
```
COCO-style annotation can be downloaded with our [downloader](#downloader) script, or
directly from here: [clipart](https://drive.google.com/uc?export=download&id=16Ra9bMXn-O5UiJ-JyN9mtf77dD1x-4lI), [comic](https://drive.google.com/uc?export=download&id=1zKuDcxvoXK8e704U10Hv2-NJvFnh0-lA) and [watercolor](https://drive.google.com/uc?export=download&id=1m7M7Vizs6uHkhHj-N6XxlgIi9zVg6q30).

#### NOTE: 
All datasets follow their original licenses.

Most of the provided ground-truth class-agnostic annotation files where taken from [CutLER](https://github.com/facebookresearch/CutLER). 

