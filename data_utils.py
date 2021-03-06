import numpy as np
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from mpl_toolkits import mplot3d
import modelnet_data
from scipy.spatial.transform import Rotation

def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc


def rotation_point_cloud(pc):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_by_angle_x(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data, rotation_matrix

def rotate_point_cloud_by_angle_y(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data, rotation_matrix

def rotate_point_cloud_by_angle_z(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data, rotation_matrix


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return:
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc += shifts
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scales = np.random.uniform(scale_low, scale_high, 1)
    pc *= scales
    return pc


def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = pc
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def pc_augment(pc, angle):
    pc = rotate_point_cloud_by_angle(pc, angle)
    # pc = rotation_point_cloud(pc)
    # pc = jitter_point_cloud(pc)
    # pc = random_scale_point_cloud(pc)
    # pc = rotate_perturbation_point_cloud(pc)
    # pc = shift_point_cloud(pc)
    return pc


def data_augment(train_data, angle):
    return pc_augment(train_data, angle).reshape(-1, train_data.shape[1], train_data.shape[2])

def npmat2euler(mats, seq='zyx'):
    # eulers = []
    # for i in range(mats.shape[0]):
    #     r = Rotation.from_dcm(mats[i])
    #     eulers.append(r.as_euler(seq, degrees=True))

    r = Rotation.from_dcm(mats.T)
    e = r.as_euler(seq, degrees=True)
    return e

def plot_pc(data, rotated, fig):

    plt.figure(fig)
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], color='Grey')
    ax.scatter3D(rotated[:, 0], rotated[:, 1], rotated[:, 2], color='Red')

    plt.show()

def plot_pc_2(data, ind):

    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], color='Grey')
    print(ind.shape[0])
    for i in range(ind.shape[0]):
        ax.scatter3D(data[ind[i], 0], data[ind[i], 1], data[ind[i], 2], color='Red')

    plt.show()

def plot_principal_axes(data,rotated, mat):
    rotated = modelnet_data.shuffle_points(rotated)
    # print(rotated.shape)
    # print(data.shape)
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], color='Grey')
    ax.scatter3D(rotated[:, 0], rotated[:, 1], rotated[:, 2], color='Red')

    plt.figure(2)

    pca = PCA(n_components=3)
    pca.fit(data)
    # data = pca.transform(data)
    pca2 = PCA(n_components=3)
    pca2.fit(rotated)
    # rotated = rotated@mat
    rotated = pca2.transform(rotated)
    # print(np.mean((mat-np.degrees(pca2.components_.T@pca.components_))*(mat-np.degrees(pca2.components_.T@pca.components_))))
    rotated = np.dot(rotated, pca.components_) #+ pca.mean_

    # print(np.degrees(pca2.components_.T@pca.components_))
    matrix = npmat2euler(pca2.components_.T@pca.components_)
    print(matrix)
    ax2 = plt.axes(projection='3d')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])


    ax2.scatter3D(data[:, 0], data[:, 1], data[:, 2], color='Grey')
    ax2.scatter3D(rotated[:, 0], rotated[:, 1], rotated[:, 2], color='Red')

    # print(data[0])
    # print(rotated[0])


    # draw_vector_3D(pca.mean_, pca.components_, pca.explained_variance_, ax2, ['Red','Green','Blue'])
    # draw_vector_3D(pca2.mean_, pca2.components_, pca2.explained_variance_, ax2, ['Green','Green','Blue'])

    # angle_x_1 = np.arccos(np.dot(np.array([1,0,0]),pca.components_[0].T)/(np.sqrt(pca.components_[0]@pca.components_[0])))
    # angle_y_1 = np.arccos(np.dot(np.array([0,1,0]),pca.components_[1].T)/(np.sqrt(pca.components_[1]@pca.components_[1])))
    # angle_z_1 = np.arccos(np.dot(np.array([0,0,1]),pca.components_[2].T)/(np.sqrt(pca.components_[2]@pca.components_[2])))
    # angle = np.arccos(np.dot(pca.components_[1],pca.components_[2].T)/(np.sqrt(pca.components_[2]@pca.components_[2])*np.sqrt(pca.components_[1]@pca.components_[1])))
    # print(angle*180/np.pi)
    # print(angle_x_1*180/np.pi)
    # print(angle_y_1*180/np.pi)
    # print(angle_z_1*180/np.pi)
    # angle_rotated = rotate_point_cloud_by_angle_x(pca.components_, angle_x_1)
    # angle_rotated = rotate_point_cloud_by_angle_y(angle_rotated, angle_x_1)
    # angle_rotated = rotate_point_cloud_by_angle_z(angle_rotated, angle_x_1)
    # draw_vector_3D(pca.mean_, angle_rotated, pca.explained_variance_, ax2, ['Red','Green','Blue'])

    plt.show()

def draw_vector_3D(x, y, variance, ax, color):

    # for i in range(1):
    #     # ax.plot([x[0]+y[i][0]*3*np.sqrt(variance[0]), x[0]], [x[1]+y[i][1]*3*np.sqrt(variance[1]), x[1]], zs=[x[2]+y[i][2]*3*np.sqrt(variance[2]), x[2]], color=color[i])
    #     ax.plot([x[0]+y[i][0], x[0]], [x[1]+y[i][1], x[1]], zs=[x[2]+y[i][2], x[2]], color=color[i])
    ax.plot([0.25, 0], [0, 0], zs=[0, 0], color='Red')
    ax.plot([0, 0], [0.25, 0], zs=[0, 0], color='Green')
    ax.plot([0, 0], [0, 0], zs=[0.25, 0], color='Blue')



    #plt.show()



