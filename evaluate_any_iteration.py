import argparse
import pickle
import modelnet_data
import pointhop
import numpy as np
import data_utils
import os
import time
import sklearn
# import matplotlib.pyplot as plt
import point_utils
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cosine
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors as kNN
import torch
from sklearn.decomposition import PCA
import open3d as o3d
import scipy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 896, 768, 640], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[256, 64, 48, 48], help='KNN query number')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_eva_temp_iter_2.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+" ")
    LOG_FOUT.flush()
    # print(out_str)

def npmat2euler(mats, deg, seq='zyx'):

    r = Rotation.from_matrix(mats)
    e = r.as_euler(seq, degrees=deg)
    return e

def transform(data, angle_x, angle_y, angle_z, translation):

    data = data_utils.jitter_point_cloud(data)

    rotation = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    data_rotated = rotation.apply(data) + translation.T

    return data_rotated

def inverse_transform(data, angle_x, angle_y, angle_z, translation):

    rotation = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    data_inverse_rotated = data - translation.T
    data_inverse_rotated = rotation.apply(data_inverse_rotated, inverse=True)

    return data_inverse_rotated

def get_salient(data, feature):
    
    new_data, centroid = point_utils.furthest_point_sample(data, 32)
    new_feature = []
    for i in range(32):
        new_feature.append(feature[centroid[i]])
    new_feature = np.array(new_feature)

    return np.squeeze(new_data), new_feature

def align(data, angle_x, angle_y, angle_z, translation):

    data_rotated = transform(data, angle_x, angle_y, angle_z, translation)

    data_e = np.expand_dims(data, axis=0)
    data_rotated_e = np.expand_dims(data_rotated, axis=0)
    data_c = np.concatenate((data_e,data_rotated_e), axis=0)

    with open(os.path.join(LOG_DIR, 'data_all_1024_896_768_640_128_64_48_48.pkl'), 'rb') as f:
        params = pickle.load(f)
    f.close()

    leaf_node_test, points = pointhop.pointhop_pred(False, data_c, pca_params=params, n_newpoint=num_point, n_sample=num_sample)
    features = np.array(leaf_node_test)
    features = features.reshape(features.shape[0],features.shape[1],features.shape[2])
    features = np.moveaxis(features,0,2)
    
    target = features[0]
    source = features[1]

    points_t = points[0]
    points_s = points[1]



    distances = sklearn.metrics.pairwise.euclidean_distances(target,source)
    # prob = scipy.special.softmax(distances,axis=0)
    pred = np.argmin(distances,axis=0)
    dist_sort = np.argsort(distances,axis=0)
    print(dist_sort[0,:10])
    print(pred[:10])
    min_dist = np.min(distances,axis=0)
    ordered = np.argsort(min_dist)
    pred = pred[ordered[:384]]
    data_x = points_s[ordered[:384]]



    sort = []
    for i in range(384):
        sort.append(points_t[pred[i]])
    data_y = np.array(sort)

    x_mean = np.mean(data_x,axis=0,keepdims=True)
    y_mean = np.mean(data_y,axis=0,keepdims=True)
    data_x = data_x - x_mean
    data_y = data_y - y_mean

    cov = (data_y.T@data_x)
    u, s, v = np.linalg.svd(cov)
    R = v.T@u.T

    if (np.linalg.det(R) < 0):
        u, s, v = np.linalg.svd(cov)
        reflect = np.eye(3)
        reflect[2,2] = -1
        v = v.T@reflect
        R = v@u.T
        # return 100000, 100000, 10000, 10000

    angle = npmat2euler(R, False)
    t = -R@y_mean.T+x_mean.T

    data_recovered = inverse_transform(data_rotated, angle[2], angle[1], angle[0], t)

    angle_pred = npmat2euler(R, True)
    angle_true = np.asarray([angle_z, angle_y, angle_x])*180/np.pi

    # print(angle_true)
    # print(angle_pred)

    mse = np.mean((angle_true-angle_pred)*(angle_true-angle_pred))
    mae = np.mean(np.abs(angle_true-angle_pred))
    mse_t = np.mean((translation-t)*(translation-t))
    mae_t = np.mean(np.abs(translation-t))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data)
    # o3d.io.write_point_cloud("sync.ply", pcd)
    # pcd_load = o3d.io.read_point_cloud("sync.ply")

    # xyz_j = data_utils.jitter_point_cloud(data_recovered)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz_j)
    # o3d.io.write_point_cloud("sync.ply", pcd)
    # pcd_load_1 = o3d.io.read_point_cloud("sync.ply")

    # pcd_load.paint_uniform_color([192/255, 0, 0])
    # pcd_load_1.paint_uniform_color([59/255, 56/255, 56/255])
    # o3d.visualization.draw_geometries([pcd_load,pcd_load_1])

    return mae, mse, mae_t, mse_t


def main():

    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    valid_data, valid_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    # new_data = []
    # new_label = []
    # for i in range(valid_label.shape[0]):
    #     if(valid_label[i]==37):
    #         new_data.append(valid_data[i])
    #         new_label.append(valid_label[i])

    # valid_data = np.array(new_data)
    # valid_label = np.array(new_label)

    mae = []
    mse = []
    mae_T = []
    mse_T = []

    for i in range(1): #valid_data.shape[0]

        # np.random.seed(i)
        data = valid_data[i]
        angle_x = np.random.uniform()*np.pi/4
        angle_y = np.random.uniform()*np.pi/4
        angle_z = np.random.uniform()*np.pi/4
        translation = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        translation = np.expand_dims(translation, axis=1)
        
        mse_r_temp = []
        mse_t_temp = []
        mae_r_temp = []
        mae_t_temp = []

        for j in range(2):

            mae_r, mse_r, mae_t, mse_t = align(data, angle_x, angle_y, angle_z, translation)
            mse_r_temp.append(mse_r)
            mse_t_temp.append(mse_t)
            mae_r_temp.append(mae_r)
            mae_t_temp.append(mae_t)
        mse_r_temp = np.array(mse_r_temp)
        mse_t_temp = np.array(mse_t_temp)
        mae_r_temp = np.array(mae_r_temp)
        mae_t_temp = np.array(mae_t_temp)
        minimum = np.argmin(mse_r_temp)

        
        mae.append(mae_r_temp[minimum])
        mse.append(mse_r_temp[minimum])
        mae_T.append(mae_t_temp[minimum])
        mse_T.append(mse_t_temp[minimum])
            
        print(i, valid_label[i], np.mean(np.array(mse)), np.sqrt(np.mean(np.array(mse))), np.mean(np.array(mae)), np.mean(np.array(mse_T)), np.sqrt(np.mean(np.array(mse_T))), np.mean(np.array(mae_T)))
        # print(i,valid_label[i], mse_r, np.sqrt(mse_r), mae_r, mse_t, np.sqrt(mse_t), mae_t)
        log_string(str(i))
        log_string(str(valid_label[i]))
        log_string(str(np.mean(np.array(mse))))
        log_string(str(np.sqrt(np.mean(np.array(mse)))))
        log_string(str(np.mean(np.array(mae))))
        log_string(str(np.mean(np.array(mse_T))))
        log_string(str(np.sqrt(np.mean(np.array(mse_T)))))
        log_string(str(np.mean(np.array(mae_T))))
        log_string("\n")



if __name__ == '__main__':
    main()