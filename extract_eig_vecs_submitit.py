import os
import submitit
import json
from pathlib import Path
# Define your inference function
import numpy as np
import argparse
from glob import glob
from extract_eig_vecs import extract_eig_vecs


class EigVecJobMapper:
    def map_jobs(self, eig_vec_dir, num_jobs, imagenet_root, split, models_batch_list, mapping_file, device):
        Path(eig_vec_dir).mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        print("Loading image files...")
        if split == "val":
            all_image_files = glob(f"{imagenet_root}/val/*.JPEG")
        elif split == "train":
            all_image_files = glob(f"{imagenet_root}/train/*/*.JPEG")
        else:
            raise ValueError(f"Invalid split {split} provided. Must be one of ['train', 'val']")
        all_images = [Path(file).name.split('.')[0] for file in all_image_files]

        existing_eig_vec_files = [set([Path(f).name.split('.')[0] for f in glob(f"{d}/*.pt")]) for d in
                                  glob(f"{eig_vec_dir}/*")]
        # make intersection of all the models files in the subdirectories to get the list of existing vectors in all models
        existing_eig_vecs = set.intersection(*existing_eig_vec_files)
        images_to_process = list(set(all_images) - set(existing_eig_vecs))
        num_files = len(images_to_process)

        if num_files == 0:
            print(f"No images left to process for split {split}")
            exit(0)

        indices = np.linspace(0, num_files, num_jobs + 1)
        indices = np.ceil(indices).astype(int)
        mapping = {
            "imagenet_root": imagenet_root,
            "split": split,
            "output_dir": eig_vec_dir,
            "models_batch_list": models_batch_list,
            "device": device,
            "job_args": {},
        }
        for i in range(len(indices) - 1):
            end_index = int(indices[i + 1])
            start_index = int(indices[i])
            mapping["job_args"][i] = {
                "start_index": start_index,
                "end_index": end_index,
                "images_to_process": images_to_process[start_index:end_index],
            }
        with open(mapping_file, "w") as f:
            json.dump(mapping, f)

    def read_extract_vecs_args(self, job_id, mapping_file):
        # Load the arguments from the arguments file
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        start_index = mapping["job_args"][str(job_id)]["start_index"]
        end_index = mapping["job_args"][str(job_id)]["end_index"]
        split = mapping["split"]
        datasets_root = mapping["imagenet_root"]
        print(
            f"Job {job_id} started. Dataset split: {split}, Dataset root: {datasets_root}, Start index: {start_index}, End index: {end_index}")
        models_batch_list = mapping["models_batch_list"]
        images_to_process = mapping["job_args"][str(job_id)]["images_to_process"]
        output_dir = mapping["output_dir"]
        device = mapping["device"]
        if split == "val":
            img_files = [os.path.join(datasets_root, "val", f"{im_name}.JPEG") for im_name in images_to_process]
        elif split == "train":
            img_files = [os.path.join(datasets_root, "train" , im_name.split('_')[0], f"{im_name}.JPEG") for im_name in
                         images_to_process]
        else:
            raise ValueError(f"Invalid split {split} provided. Must be one of ['train', 'val']")
        return {
            "img_files": img_files,
            "models_batch_list": models_batch_list,
            "output_dir": output_dir,
            "device": device
        }


def inference_job(job_id, mapping_file):
    job_mapper = EigVecJobMapper()
    extract_eig_vecs(**job_mapper.read_extract_vecs_args(job_id, mapping_file))
    return True


def submit_jobs(num_jobs, num_gpus, slurm_partition, mapping_file, job_time, jobs_dir="./submitit_jobs"):
    executor = submitit.AutoExecutor(folder=jobs_dir)
    executor.update_parameters(
        slurm_partition=slurm_partition,  # Set the Slurm partition to use
        gpus_per_node=num_gpus,  # Set the number of GPUs per node
        tasks_per_node=1,  # Set to 1 for GPU jobs
    )
    executor.parameters['time'] = 60*job_time

    jobs = []
    with executor.batch():
        for job_id in range(num_jobs):
            # Submit the job
            job = executor.submit(inference_job, job_id, mapping_file)
            jobs.append(job)

    return jobs



def get_args():
    parser = argparse.ArgumentParser("Create NCut eigenvectors for images with submitit")
    parser.add_argument("--num-jobs", type=int, default=10, help="Number of jobs to submit")
    parser.add_argument("--ngpus", type=int, default=1, help="number of gpus per node")
    parser.add_argument("--models-batch-list", nargs='+',
                        default=[("dino_s16", 256), ("dinov2_b14", 128), ("dinov2_s14", 128), ("dino_b16", 128),
                                 ("dino_s8", 20), ("dino_b8", 12)],
                        help="List of models to use. Each model is a tuple of (model_name, batch_size)")
    parser.add_argument("--out-dir", type=str, default="datasets/imagenet", help="Output directory eigen vectors")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataset-root", type=str, default="datasets/imagenet", help="Path to imagenet dataset")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--jobs-dir", type=str, default="./eig_vec_jobs", help="Directory to save jobs logs, errors, etc.")
    parser.add_argument("--slurm-partition", type=str, default="work", help="Name of the slurm partition to uses")
    parser.add_argument("--job-time", type=int, default=24, help="Number of hours to run the job")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Starting main")
    args = get_args()

    eig_vec_dir = os.path.join(args.out_dir, f"eig_vecs_{args.split}")
    Path(eig_vec_dir).mkdir(parents=True, exist_ok=True)

    mapping_file = os.path.join(args.jobs_dir, "job_mapping.json")

    job_mapper = EigVecJobMapper()
    job_mapper.map_jobs(eig_vec_dir, args.num_jobs, args.dataset_root, args.split, args.models_batch_list, mapping_file, args.device)
    jobs = submit_jobs(args.num_jobs, args.ngpus, args.slurm_partition, mapping_file, args.job_time, jobs_dir=args.jobs_dir)

    # Wait for all jobs to complete
    results = [job.result() for job in jobs]
    print(results)
