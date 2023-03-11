import csv
import time
import cv2
import os

import SSFPS_HR_Estimator
from SSFPS_HR_Estimator import FPS_HRE
from SSFPS_HR_Estimator import ConvergenceWarning_count
from SSFPS_HR_Estimator import warning_counter
import warnings
from video_realsense_file import Video
import sys
import pandas as pd
import random
import numpy as np
import argparse


def scan_all_data(root_directory, level=2):
    data_path = []
    for item in os.listdir(root_directory):
        #print('item', item)
        full_path = os.path.join(root_directory, item)
        #print(full_path)
        if os.path.isdir(full_path) and level >= 1:
            video_file = [video for video in os.listdir(full_path) if video.endswith(".bag")]
            if video_file:
                #print(video_file)
                data_path.append(video_file)
                #video_file = os.path.join(root_directory, f"{item}.avi")
            else:
                scan_all_data(full_path, level - 1)
        else:
            if full_path.endswith(".bag"):
                video_file = full_path
                #print(full_path)
                data_path.append(video_file)
    return data_path

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Japen patch rPPG estimate heart rate")
    parser.add_argument(
        "-r", "--dataset_root_path", type=str, required=True, help="video clip root path"
    )
    parser.add_argument(
        "-d", "--dataPath", type=str, required=True, help="the .csv file of video path"
    )
    parser.add_argument(
        "--save",
        default=True,
        type=bool,
        help="save result or not (default : %(default)s)",
    )
    parser.add_argument(
        "--savePath",
        default="./result/result.csv",
        type=str,
        help="the path of result (default : %(default)s)",
    )
    parser.add_argument(
        "--ma_window_size",
        default=10,
        type=int,
        help="use for windows size of moving average (default : %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default=1,
        type=int,
        help="RGB : 0 , RGB+NIR : 1 (default : %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--second",
        default=10,
        type=int,
        help="time length used for estimate HR (default : %(default)s)",
    )
    parser.add_argument(
        "-k",
        "--vote_count",
        default=500,
        type=int,
        help="count of vote (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--whiten",
        default="arbitrary-variance",
        type=str,
        help="parameter for fast ICA (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--max_iter",
        default=200,
        type=int,
        help="parameter for fast ICA (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="parallel",
        type=str,
        help="parameter for fast ICA (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--fun",
        default="cube",
        type=str,
        help="parameter for fast ICA (default: %(default)s)",
    )
    parser.add_argument(
        "--whiten_solver",
        default="eigh",
        type=str,
        help="parameter for fast ICA (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def get_random():
    return random.random(), random.random()


if __name__ == '__main__':

    start = time.time()
    args = parse_args(sys.argv[1:])
    #dataset_root_path = sys.argv[1]
    #dataset_path = sys.argv[2]

    dataset_root_path = args.dataset_root_path
    dataset_path = args.dataPath

    data_list = []
    gt_list = []
    #print(dataset_root_path)
    #print(dataset_path)
    with open(dataset_path, newline='') as csvfile:
        rows = csv.DictReader(csvfile)

        #print(rows)
        for row in rows:
            gt_list.append(row['GT'])
            #print(dataset_root_path, row['data path'])
            data_list.append(os.path.join(dataset_root_path, row['data path']))

    args = parse_args(sys.argv[1:])
    #root_directory = sys.argv[1]
    #result_path = sys.argv[2]
    warnings.simplefilter("always")
    warnings.showwarning = SSFPS_HR_Estimator.warning_counter
    root_directory = args.dataPath
    if args.save:
        result_path = args.savePath

    test_times = 1
    ica_parameter = {"whiten": args.whiten, "algorithm": args.algorithm, "fun": args.fun,
                     "whiten_solver": args.whiten_solver}

    HR_Estimator = FPS_HRE('')
    HR_Estimator.set_ma_window_size(args.ma_window_size)
    HR_Estimator.set_ica_parameter(ica_parameter)
    HR_Estimator.set_data_length(args.second)
    for i in range(len(data_list)):
        HR_Estimator.set_dataPath(data_list[i])
        bpm, fps = HR_Estimator.run()
        HR_Estimator.reset()
        if bpm is not None:
            mae = abs(bpm-float(gt_list[i]))
        else:
            mea = bpm

        print('gt : ', float(gt_list[i]))
        data = {"data path": data_list[i] , "bpm": bpm, "fps": fps, "MAE": mae,
                'convergenceWarning_count': ConvergenceWarning_count}

        df = pd.DataFrame(data, index=[0])
        #with open(result_path, "a") as f:
        #    df.to_csv(f, header=True, index=False)
        df.to_csv(result_path, header=False, index=False, mode='a')

    end = time.time()
    print("all run time : ", (end-start)/60)
