""" logs and tensorboard viz to be kept are recorded here, when running the script it will delete all other run outputs that are not recorded here
"""
import os, glob
import shutil
keep_runs = [
    "outputs/2021-03-18/15-20-36",
    "outputs/2021-03-18/17-05-56",
    "outputs/2021-03-18/21-11-07",
    "outputs/2021-03-18/19-44-16",
    "outputs/2021-03-18/20-23-58",
    "outputs/2021-03-18/21-01-00",
    "outputs/2021-03-19/10-00-21",
]


if __name__ == "__main__":
    output_dirs = sorted(glob.glob("outputs" + '/*/*', recursive=True))
    for dir in output_dirs:
        if not dir in keep_runs:
            # shutil.rmtree(dir)