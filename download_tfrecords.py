import os
from pathlib import Path
import subprocess
import argparse
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():

    parser = argparse.ArgumentParser(description='Waymo Downloader')
    parser.add_argument('--root_path', type=str, required=True) # Main Data directory
    parser.add_argument('--split', type=str, required=True)
    
    args = parser.parse_args()

    return args


def get_download_list(split):
    ''' check connection with google cloud and get the full download list for the requested split
    '''
    print("Python Version is {}.{}.{}".format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    if sys.version_info[1] <= 6:
        from subprocess import PIPE
        p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{type}".format(split), shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    else:
        p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{}".format(split), shell=True, capture_output=True, text=True)
    
    if p1.returncode != 0:
        raise Exception("gsutil error, check that gsutil is installed and configured correctly according to README file !", p1.stderr)
    
    download_list = p1.stdout.split()
    return download_list[1::] # skip first entry, it contains cloud folder path

def download_tf_record(download_link, SPLIT_PATH):
    p1 = subprocess.run(" ".join(["gsutil cp", download_link, SPLIT_PATH]), shell=True)
    if p1.returncode != 0:
        raise Exception('problem with download cmd')
    tf_record_file_name = download_link.split('/')[-1] # get file name
    tf_record_file_path = os.path.join(SPLIT_PATH, tf_record_file_name)

    return tf_record_file_path

def main (args):
    
    SPLIT_PATH = os.path.join(args.root_path, 'Waymo', 'tfrecord_{}'.format(args.split))
    Path(SPLIT_PATH).mkdir(parents=True, exist_ok=True)

    segments_download_links_list = get_download_list(args.split)
    
    for segment_link in segments_download_links_list:
        download_tf_record(segment_link, SPLIT_PATH)

if __name__ == '__main__':
    
    args = parse_args()
    main(args)