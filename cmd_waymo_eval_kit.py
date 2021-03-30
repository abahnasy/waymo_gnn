""" launch waymo evaluation kit for detection
"""
#TODO: later, provide predictions and gt files as input, now hardcoded in the file

import os

import fire


def evaluate_detection():
    cmd = ("/home/bahnasy/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main /home/bahnasy/waymo_gnn/ckpts/epoch_36/detection_pred.bin /home/bahnasy/waymo_gnn/data/Waymo/gt_preds.bin")
    print(cmd)
    os.system(cmd)

def evaluate_tracking():
    cmd = ("/home/bahnasy/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main /home/bahnasy/waymo_gnn/output_tracking/tracking_pred.bin /home/bahnasy/waymo_gnn/data/Waymo/gt_preds.bin")
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    fire.Fire()



