import os
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
from PIL import Image
from networks import get_model
from torch.utils.data import DataLoader, Dataset
import time
from votecut.ncut import ncut


# Image transformation applied to all images
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ImagesDataset(Dataset):
    def __init__(self, images_files, transform=None, resize=(480, 480)):
        self.images_files = images_files
        self.transform = transform
        self.resize = resize

    def __getitem__(self, index):
        image_file = self.images_files[index]

        img = Image.open(image_file).convert('RGB')
        img = img.resize(self.resize, Image.Resampling.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        return img, image_file

    def __len__(self):
        return len(self.images_files)

    @property
    def images_size(self):
        return self.resize


def save_batch_eig_vecs(batch_files, eigenvectors, save_dir):
    for i, file in enumerate(batch_files):
        file_name = file.split("/")[-1].split(".")[0]
        torch.save(eigenvectors[i].detach().cpu().clone(), os.path.join(save_dir, f"{file_name}.pt"))


def get_images_imagenet_files(imagenet_root, split):
    print("Loading image files...")
    if split == "val":
        image_files = glob(f"{imagenet_root}/val/*.JPEG")
    elif split == "train":
        image_files = glob(f"{imagenet_root}/train/*/*.JPEG")
    else:
        raise ValueError(f"Invalid split {split} provided. Must be one of ['train', 'val']")
    return image_files


def extract_eig_vecs(img_files, models_batch_list, output_dir, device="cuda"):
    for model_name, batch_size in models_batch_list:
        # read the integer patch size from the model name
        model, patch_size = get_model(model_name, device)
        save_dir = f"{output_dir}/{model_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if "dinov2" in model_name:
            dataset = ImagesDataset(img_files, transform=transform, resize=(476, 476))
        else:
            dataset = ImagesDataset(img_files, transform=transform)
        ts = time.time()
        model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        token_num = (dataset.images_size[0]//patch_size)**2
        with torch.no_grad():
            for i, (batch_images, batch_files) in enumerate(tqdm(data_loader, desc=f"Processing {model_name} model")):
                batch_images = batch_images.to(device)
                # to follow prior work we take the "key" features if the model is DINO, for DINOv2 we take the last layer
                if "dinov2" in model_name:
                    features = model(batch_images, return_patches=True)
                else:
                    _, k, _ = model.get_last_qkv(batch_images)
                    k = k.transpose(1, 2).reshape(batch_images.shape[0], token_num + 1, -1)
                    features = k[:, 1:, :]
                eigenvectors, eigenvalues = ncut(features, tau=0.15)
                save_batch_eig_vecs(batch_files, eigenvectors, save_dir)
                del features
        del model
        te = time.time()
        inference_time = te - ts
        print(f"Total inference time: {inference_time}, model: {model_name}")


if __name__ == "__main__":
    # anns = json.load(open("/data/home/ssaricha/TokenCut/results/tmp_res_v1_K_6_eig_6_filter_small_post_iou_05_limit_10_with_mae_mask_thresh_010_COCO_17_val/ann_v1_K_6_eig_6_filter_small_post_iou_05_limit_10_with_mae_mask_thresh_010_COCO_17_pl_299.json"))
    parser = argparse.ArgumentParser("Create eigenvectors for all models")
    parser.add_argument("--dataset-root", type=str, default="datasets/imagenet", help="Path to coco dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--output-dir", type=str, default="datasets/eig_vecs_val", help="Output directory for json results evaluation file")
    parser.add_argument("--device", type=str, default="cuda", help="computation device", choices=["cpu", "cuda"])
    # add argument of list of models to use
    parser.add_argument("--models-batch-list", nargs='+',
                        default=[("dino_s16", 512), ("dinov2_b14", 256), ("dinov2_s14", 256), ("dino_b16", 256), ("dino_s8", 32), ("dino_b8", 16)],
                        help="List of models to use. Each model is a tuple of (model_name, batch_size)")
    args = parser.parse_args()

    device = args.device
    dataset_root = args.dataset_root
    if args.split == "val":
        img_files = glob(f"{dataset_root}/val/*.JPEG")
    elif args.split == "train":
        img_files = glob(f"{dataset_root}/train/*/*.JPEG")
    else:
        raise ValueError(f"Invalid split {args.split} provided. Must be one of ['train', 'val']")
    models_batch_list = args.models_batch_list
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extract_eig_vecs(img_files, models_batch_list, output_dir, device)

    exit(0)

