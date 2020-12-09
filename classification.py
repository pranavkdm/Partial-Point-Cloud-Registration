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
parser.add_argument('--num_point', default=[1024, 512, 256, 64], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[32, 32, 32, 32], help='KNN query number')
parser.add_argument('--threshold', default=0.001, help='threshold')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
threshold = FLAGS.threshold
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():

    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    valid_data, valid_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    with open(os.path.join(LOG_DIR, 'data_all.pkl'), 'rb') as f:
        params = pickle.load(f)
        f.close()

    leaf_node_train, points_train = pointhop.pointhop_pred(False, train_data, pca_params=params, n_newpoint=num_point, n_sample=num_sample)
    leaf_node_test, points_test = pointhop.pointhop_pred(False, valid_data, pca_params=params, n_newpoint=num_point, n_sample=num_sample)

    feature_train = pointhop.extract(leaf_node_train)
    feature_train = np.concatenate(feature_train, axis=-1)

    feature_valid = pointhop.extract(leaf_node_test)
    feature_valid = np.concatenate(feature_valid, axis=-1)

    weight = pointhop.llsr_train(feature_train, train_label)
    feature_train, pred_train = pointhop.llsr_pred(feature_train, weight)
    acc_train = sklearn.metrics.accuracy_score(train_label, pred_train)
    print("Train accuracy is - ", acc_train, " %")
    feature_valid, pred_valid = pointhop.llsr_pred(feature_valid, weight)
    acc_test = sklearn.metrics.accuracy_score(valid_label, pred_valid)
    print("Test accuracy is - ", acc_test, " %")

if __name__ == '__main__':
    main()
