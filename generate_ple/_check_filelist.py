import os
import pickle
import numpy as np
import os.path as osp
from tqdm import tqdm
from argparse import ArgumentParser

# check info file


if __name__ == '__main__':
    base_path = '/raid/local/cvml_user/seun/dataset/SemanticKITTI2/dataset/'
    labeled_name = 'semantickitti_infos_train.0.5.pkl'

    labeled_pkl = pickle.load(open(os.path.join(base_path, labeled_name), "rb"))
    labeled_datalist = labeled_pkl['data_list']
    print(os.path.join(base_path, labeled_name))
    for i in range(10):
        lidar_path = labeled_datalist[i]['lidar_points']['lidar_path']
        print(lidar_path)