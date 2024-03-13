import os
import argparse
import json
from glob import glob
from utils.annotaions_worker import CocoAnnotationsWorker
from create_pseudo_masks import create_votecut_annotations
import numpy as np
from pathlib import Path
import uuid
import submitit


class AnnJobMapper:
    """
    This class is responsible for mapping the jobs to the workers. It creates and reads the mapping file to get the
     arguments for the workers method.
    """
    def map_jobs(self, eig_vec_dirs, Ks, tau_m, num_eig_vecs, num_jobs, save_period, jobs_dir,
                 imagenet_root, split, mapping_file, device, resume):
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        print("Mapping image files...")
        if split == "val":
            all_image_files = glob(f"{imagenet_root}/val/*.JPEG")
        elif split == "train":
            all_image_files = glob(f"{imagenet_root}/train/*/*.JPEG")
        else:
            raise ValueError(f"Invalid split {split} provided. Must be one of ['train', 'val']")

        # if resume, remove the files that have already been processed
        if resume:
            existing_files = self.existing_files(jobs_dir)
            all_image_files = set(all_image_files) - set(existing_files)

        run_dir = os.path.join(jobs_dir, "run-" + uuid.uuid4().hex[:10])
        num_files = len(all_image_files)
        indices = np.linspace(0, num_files, num_jobs + 1)
        indices = np.ceil(indices).astype(int)
        mapping = {
            "imagenet_root": imagenet_root,
            "split": split,
            "eig_vec_dirs": eig_vec_dirs,
            "Ks": Ks,
            "tau_m": tau_m,
            "num_eig_vecs": num_eig_vecs,
            "save_period": save_period,
            "device": device,
            "job_args": {},
        }
        jobs_to_run = []
        for i in range(len(indices) - 1):
            worker_dir = os.path.join(run_dir, f"job_{i}")
            job_worker = CocoAnnotationsWorker(worker_dir)
            if resume and job_worker.is_done():
                continue
            jobs_to_run.append(i)
            end_index = int(indices[i + 1])
            start_index = int(indices[i])
            mapping["job_args"][i] = {
                "start_index": start_index,
                "end_index": end_index,
                "images_to_process": all_image_files[start_index:end_index],
                "resume": False, # we don't resume in the worker itself since we are resuming here by remapping the jobs
                "worker_dir": worker_dir
            }
        if len(jobs_to_run) > 0:
            print(f"Mapping file created: {mapping_file}")
            with open(mapping_file, "w") as f:
                json.dump(mapping, f)
        return jobs_to_run

    def existing_files(self, jobs_dir):
        # get all the images that have been processed
        all_worker_ann_files = glob(os.path.join(jobs_dir, "*", "ann_*.json"))
        existing_files = []
        for ann_file in all_worker_ann_files:
            with open(ann_file, "r") as f:
                tmp_anns = json.load(f)
                existing_files.extend([a["file_name"] for a in tmp_anns])
        return existing_files

    def read_create_annotations_args(self, job_id, mapping_file):
        # Load the arguments from the arguments file
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        print(
            f"Job {job_id} started. Dataset root: {mapping['imagenet_root']}, Start index: {mapping['job_args'][str(job_id)]['start_index']}, End index: {mapping['job_args'][str(job_id)]['end_index']}")
        return {
            "eigenvec_dirs": mapping["eig_vec_dirs"],
            "img_files": mapping["job_args"][str(job_id)]["images_to_process"],
            "Ks": mapping["Ks"],
            "tau_m": mapping["tau_m"],
            "num_eig_vecs": mapping["num_eig_vecs"],
            "save_period": mapping["save_period"],
            "worker_dir": mapping["job_args"][str(job_id)]["worker_dir"],
            "device": mapping["device"],
            "resume": mapping["job_args"][str(job_id)]["resume"]
        }


def annotations_job(job_id, mapping_file):
    job_mapper = AnnJobMapper()
    create_votecut_annotations(**job_mapper.read_create_annotations_args(job_id, mapping_file))
    return True


def submit_jobs(jobs_to_run, num_gpus, slurm_partition, mapping_file, job_time, jobs_dir="./submitit_jobs"):
    executor = submitit.AutoExecutor(folder=jobs_dir)
    executor.update_parameters(
        slurm_partition=slurm_partition,
        gpus_per_node=num_gpus,
        tasks_per_node=1,
    )
    executor.parameters['time'] = 60*job_time

    jobs = []
    with executor.batch():
        for job_id in jobs_to_run:
            # Submit the job
            job = executor.submit(annotations_job, job_id, mapping_file)
            jobs.append(job)

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file with submitit. This script also \
                                     supports resuming from previous runs to handle the case of job failure. Different \
                                     launches of this script should have different --jobs-dir and --out-dir to avoid \
                                     conflicts.")

    parser.add_argument("--dataset-root", type=str, default="datasets/imagenet",
                        help="Path to imagenet dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to use")
    parser.add_argument("--Ks", type=tuple, default=(2, 3), help="Ks to use for kmeans")
    parser.add_argument("--out-dir", type=str, default="datasets/imagenet/annotations", help="")
    parser.add_argument("--tau-m", type=float, default=0.2, help="")
    parser.add_argument("--models", nargs='+',
                        default=["dino_s16", "dinov2_b14", "dinov2_s14", "dino_b16", "dino_s8", "dino_b8"],
                        help="List of models to use")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")

    parser.add_argument("--num-jobs", type=int, default=100, help="Number of jobs to submit")
    parser.add_argument("--ngpus", type=int, default=0,
                        help="Number of gpus per node. Note: this task not parallelizable on gpus, so set to 0. \
                         Increase the number of jobs to parallelize instead")
    parser.add_argument("--slurm-partition", type=str, default="work",
                        help="Name of the slurm partition to uses")
    parser.add_argument("--job-time", type=int, default=24, help="Number of hours to run the job")
    parser.add_argument("--save-tmp-files", action="store_true",
                        help="Save temp files even all jobs are done")
    parser.add_argument("--num-eig-vecs", type=int, default=1, help="Number of eigen vectors to use")
    parser.add_argument("--save-period", type=int, default=100,
                        help="Saving period for the annotations in temp files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs-dir", type=str, default="./.masks_jobs",
                        help="Directory to save jobs logs, errors, etc.")


    args = parser.parse_args()

    eig_vec_dirs = [os.path.join(args.dataset_root, f"eig_vecs_{args.split}", model) for model in args.models]

    mapping_file = os.path.join(args.jobs_dir, "job_mapping.json")

    print(f"Submitting {args.num_jobs} jobs to create pseudo labels for {args.split} split")
    print(f"Mapping jobs...")
    job_mapper = AnnJobMapper()
    # map jobs to workers
    jobs_to_run = job_mapper.map_jobs(eig_vec_dirs, args.Ks, args.tau_m, args.num_eig_vecs, args.num_jobs,
                                      args.save_period, args.jobs_dir, args.dataset_root, args.split, mapping_file,
                                      args.device, args.resume)
    if len(jobs_to_run) == 0:
        print(f"No jobs to run. All jobs are already done.")
        results = [True]
    else:
        print(f"Submitting jobs...")
        jobs = submit_jobs(jobs_to_run, args.ngpus, args.slurm_partition, mapping_file, args.job_time,
                           jobs_dir=args.jobs_dir)
        print(f"Jobs submitted. Waiting for jobs to complete...")
        # Wait for all jobs to complete
        results = [job.result() for job in jobs]
    if all(results):
        print("Aggregating all annotations...")
        # the tmp files are in the format: <jobs_dir>/run-<uuid>/job_<job_id>/ann_<job_id>.json
        all_tmp_ann_files = glob(f"{args.jobs_dir}/*/*/*.json")
        # aggregate all the annotations and save to a single file
        anns = CocoAnnotationsWorker.collect_to_single_ann_dict(all_tmp_ann_files, imgnet_train=args.split == "train")
        ann_file = f"imagenet_{args.split}_votecut_kmax_{max(args.Ks)}_tuam_{args.tau_m}.json"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(args.out_dir, ann_file)
        print(f"Saving aggregated annotations to {out_file}")
        with open(out_file, "w") as f:
            json.dump(anns, f)
        if not args.save_tmp_files:
            print("Cleaning up temp files...")
            CocoAnnotationsWorker.cleanup_tmp_files(args.jobs_dir)
        print("Done!")
    else:
        print("Some jobs failed. Please check logs for more details")

