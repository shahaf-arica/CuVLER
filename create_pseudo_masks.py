import os
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image
import time
from votecut.votecut import votecut
from utils.annotaions_worker import CocoAnnotationsWorker
import json



def load_eig_vecs(eigenvec_dirs, num_eig_vecs, image_name):
    """
    Load the eigen vectors for the image in the format of a dictionary votecut method expects
    :param eigenvec_dirs: list of directories containing the eigen vectors
    :param num_eig_vecs: number of eigen vectors to use from each directory
    :param image_name: name of the image without the extension
    :return:
    """
    # load eigen vectors
    vector_groups = {}
    for eigenvec_dir in eigenvec_dirs:
        eig_vec_path = os.path.join(eigenvec_dir, f"{image_name}.pt")
        eig_vec = torch.load(eig_vec_path)
        eig_vec = eig_vec.T
        eig_vec = eig_vec[:num_eig_vecs]
        if eig_vec.shape[1] == 900:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 30, 30)
        elif eig_vec.shape[1] == 3600:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 60, 60)
        elif eig_vec.shape[1] == 1156:
            eig_vec = eig_vec.reshape(eig_vec.shape[0], 34, 34)
        else:
            raise ValueError("Invalid eig vec shape")
        vector_groups[eigenvec_dir] = {
            "eigenvectors": eig_vec
        }
    return vector_groups


def parse_image_file(image_full_path):
    image_name = Path(image_full_path).stem.split(".")[0]
    # in case the file name starts with ILSVRC2012 remove it, it is the validation prefix
    image_id = image_name[len("ILSVRC2012"):] if image_name.startswith("ILSVRC2012") else image_name
    image_id = int("".join(filter(str.isdigit, image_id)))
    return image_name, image_id


def create_votecut_annotations(eigenvec_dirs, img_files, Ks, worker_dir,
                               tau_m=0.2, num_eig_vecs=1, save_period=100, device="cpu", resume=False):
    """
    This is a method for a single job that creates the pseudo labels for the images using votecut method and save them
    to a temporary file in order to be aggregated later. That way we can parallelize the process of creating the pseudo
    labels for the images, and also saving RAM by not keeping all the annotations in memory.
    :param eigenvec_dirs: list of directories containing the eigen vectors
    :param img_files: list of image files to process
    :param Ks: Ks to use for kmeans
    :param worker_dir: directory to save the temporary files
    :param tau_m: tau_m to use for votecut
    :param num_eig_vecs: number of eigen vectors to use
    :param save_period: saving period for the annotations in temp files
    :param device:
    :param resume:
    :return:
    """
    ts = time.time()
    ann_worker = CocoAnnotationsWorker(worker_dir)
    # if the worker directory exists and we are not resuming the process clear it
    if resume:
        img_files = ann_worker.resume(img_files)
    else:
        ann_worker.cleanup()

    if len(img_files) == 0:
        print("No images left to process, exiting...")
        return

    Path(worker_dir).mkdir(parents=True, exist_ok=True)
    dataset_pseudo_labels_dicts = []
    # just for tracking the skipped images
    skipped_images_file = os.path.join(worker_dir, "skipped_images.txt")


    for ind, img_file in enumerate(tqdm(img_files, desc="Creating pseudo labels")):
        try:
            image_name, image_id = parse_image_file(img_file)
            # load all eigen vectors for the image
            image_rgb = Image.open(img_file).convert("RGB")
            # load all eigen vectors for the image
            eig_vec_groups = load_eig_vecs(eigenvec_dirs, num_eig_vecs ,image_name)
            # perform votecut on the image
            image_masks = votecut(image_rgb, eig_vec_groups, Ks=Ks, tau_m=tau_m, device=device)
            # add the image annotations
            success = ann_worker.add_image_ann(image_id=image_id,
                                               file_name=img_file,
                                               height=image_rgb.size[1],
                                               width=image_rgb.size[0],
                                               image_masks=image_masks)
            # write the image file to the existing files
            if not success:
                print(f"Failed to add image {img_file} to the annotations")
                with open(skipped_images_file, "a") as f:
                    f.write(f"{image_name}\n")
                continue
        except Exception as e:
            print(f"Error: {e}")
        # save the annotations to temp file for aggregation
        if (ind + 1) % save_period == 0:
            ann_worker.flush_and_save_anns()
    # save the leftover annotations
    if len(dataset_pseudo_labels_dicts) > 0:
        ann_worker.flush_and_save_anns()
    ann_worker.done()
    te = time.time()
    print(f"Running Time: {te - ts}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file")
    parser.add_argument("--dataset-root", type=str, default="datasets/imagenet", help="Path to coco dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to use")
    parser.add_argument("--Ks", type=tuple, default=(2,3), help="Ks to use for kmeans")
    parser.add_argument("--out-file", type=str, default="annotations.json", help="")
    parser.add_argument("--tau-m", type=float, default=0.2, help="")
    parser.add_argument("--models", nargs='+',
                        default=["dino_s16", "dinov2_b14", "dinov2_s14", "dino_b16", "dino_s8", "dino_b8"],
                        help="List of models to use")
    parser.add_argument("--eig-vec-dir", type=str, default="datasets/eig_vecs_val", help="Directory of images eigen vectors for each model")
    parser.add_argument("--num-eig-vecs", type=int, default=1, help="Number of eigen vectors to use")
    parser.add_argument("--save-period", type=int, default=100, help="saving period for the annotations in temp files")
    parser.add_argument("--tmp-folder", type=str, default="tmp", help="Directory to save temp files")
    parser.add_argument("--save-tmp-files", action="store_true", help="Save temp files")
    parser.add_argument("--resume", type=bool, default=False, help="Resume from previous run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    Ks = args.Ks
    tmp_folder = args.tmp_folder
    eigenvec_dirs = [f"{args.eig_vec_dir}/{model}" for model in args.models]
    if args.split == "val":
        all_image_files = glob(f"{args.dataset_root}/val/*.JPEG")
    elif args.split == "train":
        all_image_files = glob(f"{args.dataset_root}/train/*/*.JPEG")
    else:
        raise ValueError(f"Invalid split {args.split} provided. Must be one of ['train', 'val']")
    create_votecut_annotations(eigenvec_dirs, all_image_files, args.Ks, tmp_folder, args.tau_m, args.num_eig_vecs, args.save_period, args.device, args.resume)
    anns = CocoAnnotationsWorker.collect_to_single_ann_dict(tmp_folder, args.out_file)
    with open(args.out_file, "w") as f:
        json.dump(anns, f)
    if not args.save_tmp_files:
        CocoAnnotationsWorker.cleanup_tmp_files(tmp_folder)
    exit(0)
