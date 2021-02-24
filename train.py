import argparse
import pickle
import modelnet_data
import pointhop
import numpy as np
import data_utils
import os
import time
import sklearn
import h5py
import point_utils
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--log_dir', default='./log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 896, 768, 640], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[128, 64, 48, 48], help='KNN query number')
parser.add_argument('--threshold', default=0.001, help='threshold')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
threshold = FLAGS.threshold
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():

    BASE_DIR = '/mnt/pranav2/'

    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    valid_data, valid_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    params, leaf_node, leaf_node_energy = pointhop.pointhop_train(True, train_data, n_newpoint=num_point,
                                        n_sample=num_sample, threshold=threshold)

    with open(os.path.join(LOG_DIR, 'data_all_1024_896_768_640_128_64_48_48.pkl'), 'wb') as f:
        pickle.dump(params, f)

if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    print(time.time()-t0)