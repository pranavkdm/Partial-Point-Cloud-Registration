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
parser.add_argument('--validation', default=True, help='Split train data or not')
parser.add_argument('--ensemble', default=False, help='Ensemble or not')
parser.add_argument('--rotation_angle', default=np.pi/4, help='Rotate angle')
parser.add_argument('--rotation_freq', default=8, help='Rotate time')
parser.add_argument('--log_dir', default='./log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 512, 256, 64], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[32, 32, 32, 32], help='KNN query number')
parser.add_argument('--threshold', default=0.001, help='threshold')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
VALID = FLAGS.validation
ENSEMBLE = FLAGS.ensemble
angle_rotation = FLAGS.rotation_angle
freq_rotation = FLAGS.rotation_freq
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
    # load data

    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    valid_data, valid_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    # new_data = []
    # for i in range(train_label.shape[0]):
    #     if(train_label[i]<20):
    #         new_data.append(train_data[i])
    # train_data = np.array(new_data)

    train_data = train_data[:500]
    train_label = train_label[:500]
    valid_data = valid_data[:500]
    valid_label = valid_label[:500]

    # params, leaf_node, leaf_node_energy = pointhop.pointhop_train(True, train_data, n_newpoint=num_point,
    #                                     n_sample=num_sample, threshold=threshold)

    with open(os.path.join(LOG_DIR, 'data_all.pkl'), 'rb') as f:
        params = pickle.load(f)
        f.close()

    leaf_node_train, points_train = pointhop.pointhop_pred(False, train_data, pca_params=params, n_newpoint=num_point, n_sample=num_sample)
    leaf_node_test, points_test = pointhop.pointhop_pred(False, valid_data, pca_params=params, n_newpoint=num_point, n_sample=num_sample)

    feature_train = pointhop.extract(leaf_node_train)
    feature_train = np.concatenate(feature_train, axis=-1)
    print(feature_train.shape)

    feature_valid = pointhop.extract(leaf_node_test)
    feature_valid = np.concatenate(feature_valid, axis=-1)
    print(feature_valid.shape)

    weight = pointhop.llsr_train(feature_train, train_label)
    feature_train, pred_train = pointhop.llsr_pred(feature_train, weight)
    acc_train = sklearn.metrics.accuracy_score(train_label, pred_train)
    print(acc_train)
    feature_valid, pred_valid = pointhop.llsr_pred(feature_valid, weight)
    acc_test = sklearn.metrics.accuracy_score(valid_label, pred_valid)
    print(acc_test)

    # with open(os.path.join(LOG_DIR, 'data_all.pkl'), 'wb') as f:
    #     pickle.dump(params, f)

if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    print(time.time()-t0)