#!/usr/bin/env python3
"""
Build a LeRobot dataset from post_process.py output folders and (optionally) push it to the Hugging Face Hub.
The episode format is described in dataset_io.py.

    uv run python packages/data/log.py --overwrite                  # rebuild from every output/<recording>/
    uv run python packages/data/log.py --append output/demo23_nov24 # add these folders to the existing dataset
    uv run python packages/data/log.py --push                       # upload the dataset as it is (private)
    uv run python packages/data/log.py --overwrite --push --public  # options combine

A recording already in the dataset (meta/sources.json) is skipped, so --append is safe to repeat.
Uploading needs a Hugging Face login once:  uv run hf auth login   (paste a token with write access)
"""
import argparse
import glob
import os
import shutil
import sys

from dataset_io import DEFAULT_REPO_ID, EpisodeError, EpisodeWriter, check_hub_login


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true",
                      help="delete the local dataset and rebuild it from every folder under --input")
    mode.add_argument("--append", nargs="+", metavar="FOLDER",
                      help="add these post_process output folders to the dataset (created if missing)")
    ap.add_argument("--input", default="output", help="folder with one sub-folder per recording (default: output/)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--root", default=None, help="local dataset folder (default: datasets/<repo name>)")
    ap.add_argument("--task", default=None,
                    help="task description stored with every frame (default: 'put the <object 0> in the <object 1>')")
    ap.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub at the end")
    ap.add_argument("--public", action="store_true", help="make the Hub dataset public (default: private)")
    args = ap.parse_args()
    if not (args.overwrite or args.append or args.push):
        ap.error("nothing to do: pass --overwrite, --append FOLDER... and/or --push")

    root = args.root or os.path.join("datasets", args.repo_id.split("/")[-1])
    if args.push:
        check_hub_login()

    if args.overwrite:
        folders = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.input, "*", "*_aina.npz")))
        if not folders:
            sys.exit(f"no <recording>/<recording>_aina.npz found under {args.input}/")
        print(f"found {len(folders)} recording(s) under {args.input}/")
        if os.path.exists(root):
            shutil.rmtree(root)
    else:
        folders = args.append or []

    if folders:
        try:
            writer = EpisodeWriter(root, args.repo_id, args.task)
        except EpisodeError as e:
            sys.exit(str(e))
        added = failed = 0
        with writer:
            for folder in folders:
                try:
                    added += writer.add(folder)
                except EpisodeError as e:
                    failed += 1
                    print(f"  skipping: {e}")
        print(f"{root}: added {added}, skipped {failed} with errors, {len(writer.sources)} episodes in total")
        if not writer.sources:
            sys.exit("the dataset is empty, nothing to push")

    if args.push:
        if not os.path.isdir(root):
            sys.exit(f"{root} does not exist")
        EpisodeWriter(root, args.repo_id).push(private=not args.public)


if __name__ == "__main__":
    main()
