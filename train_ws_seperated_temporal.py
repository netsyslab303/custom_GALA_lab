import copy
import math

import keras.models
import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import os, time, argparse
import pickle
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import op_util
from nets import GALA

import tensorflow as tf
import torch
from tensorflow import keras

normalized_point = 0
num_adj = 1
batch = 64
time_input = 30  # 입력으로 사용되는 frame 수
frame_interval = 15  # Pkl 파일에서의 입력 간격 (Ex. frame :0 ~ 100 있는 PKL, time_input 30, frame_interval 15 => 0~30/ 15~45/ 30~60/ 45~75 ...
num_noise = 10

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="test", type=str)
parser.add_argument("--save_model_name",
                    default="weights/t_t_s_{}_adj{}_{}_{}_{}".format(normalized_point, num_adj, time_input,
                                                                               frame_interval, num_noise), type=str)
parser.add_argument("--load_model", action='store_true')
parser.add_argument("--test", action='store_false', help='for only_test')
args = parser.parse_args()
# 입력으로 사용되는 frame 중 noise 처리되는 frame 수

neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                 (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                 (1, 0), (3, 1), (2, 0), (4, 2)]


def make_noise(j_feature, b_feature):
    noised_j_out = copy.deepcopy(j_feature)
    noised_b_out = copy.deepcopy(b_feature)
    n_frame = []
    n_joint = []
    for batch_idx in range(len(j_feature)):
        # 동일 time_input 안 에서는 동일한 관절만 0으로 noising
        rnd_frame = random.sample(range(0, time_input), num_noise)
        rnd_joint = random.randint(1, 16)
        n_frame.append(rnd_frame)
        n_joint.append(rnd_joint)
        # zero = [rnd_frame[i] * 17 + rnd_joint for i in range(len(rnd_frame))]
        # noised_out[batch_idx][zero,:] = 0
        for i, j in neighbor_link:
            if i == rnd_joint:
                neighbor_joint = j
                break
        noised_point = [rnd_frame[i] * 17 + rnd_joint for i in range(len(rnd_frame))]
        neighbor_point = [rnd_frame[i] * 17 + neighbor_joint for i in range(len(rnd_frame))]
        noised_j_out[batch_idx][noised_point, :] = noised_j_out[batch_idx][neighbor_point, :]

        for i, j in neighbor_link:
            if j == rnd_joint:
                neighbor_joint = i
                noised_point = [rnd_frame[i] * 17 + rnd_joint for i in range(len(rnd_frame))]
                neighbor_point = [rnd_frame[i] * 17 + neighbor_joint for i in range(len(rnd_frame))]
                noised_b_out[batch_idx][neighbor_point, :] = (noised_j_out[batch_idx][neighbor_point, :] -
                                                      noised_j_out[batch_idx][noised_point, :])
            elif i == rnd_joint:
                neighbor_joint = j
                noised_point = [rnd_frame[i] * 17 + rnd_joint for i in range(len(rnd_frame))]
                neighbor_point = [rnd_frame[i] * 17 + neighbor_joint for i in range(len(rnd_frame))]
                noised_b_out[batch_idx][noised_point, :] = (noised_j_out[batch_idx][noised_point, :] -
                                                      noised_j_out[batch_idx][neighbor_point, :])
    return noised_j_out.astype(np.float32), noised_b_out.astype(np.float32), n_frame, n_joint



def making_skeleton_adj():
    # 17*time_input 크기의 큰 인접행렬 생성
    adj_mat = np.zeros((17, 17))
    zero_mat = np.zeros((17, 17))
    idt_mat = np.eye(17)

    for i, j in neighbor_link:
        adj_mat[i][j] = 1

    init_adj_mat = [[zero_mat] * time_input for _ in range(time_input)]
    init_idt_mat = [[zero_mat] * time_input for _ in range(time_input)]
    for i in range(time_input):
        init_adj_mat[i][i] = copy.deepcopy(adj_mat)
        if num_adj == 1:
            if i < time_input - 1:
                init_idt_mat[i][i + 1] = idt_mat
        elif num_adj == 2:
            if i < time_input - 2:
                init_idt_mat[i][i + 1] = idt_mat
                init_idt_mat[i][i + 2] = idt_mat
            elif i < time_input - 1:
                init_idt_mat[i][i + 1] = idt_mat

    idt_result = []
    adj_result = []
    for i in range(time_input):
        idt_result.append(np.reshape(np.array(init_idt_mat[i]), (17 * time_input, 17)).T)
        adj_result.append(np.reshape(np.array(init_adj_mat[i]), (17 * time_input, 17)).T)
    idt_mat_out = np.concatenate(idt_result)
    adj_mat_out = np.concatenate(adj_result)

    idj_csr = ss.csr_matrix(idt_mat_out)
    adj_csr = ss.csr_matrix(adj_mat_out)

    return adj_csr, idj_csr


def get_affinity_skeleton(A, domain):
    A.toarray()
    A = A * np.exp(-1)
    A = A + A.T
    A = ss.csr_matrix.todense(A)
    eye = np.eye(A.shape[0])

    Asm = A + eye
    tmp = np.squeeze(np.asarray(Asm))
    Dsm = 1 / np.sqrt(np.sum(Asm, -1))
    DADsm = ss.csr_matrix(np.multiply(np.multiply(Asm, Dsm).T, Dsm).reshape(-1))

    Asp = 2 * eye - A
    Dsp = 1 / np.sqrt(np.sum(2 * eye + A, -1))
    DADsp = ss.csr_matrix(np.multiply(np.multiply(Asp, Dsp).T, Dsp).reshape(-1))

    DADsm_indices = np.vstack([DADsm.indices // A.shape[0], DADsm.indices % A.shape[0]]).T
    DADsp_indices = np.vstack([DADsp.indices // A.shape[0], DADsp.indices % A.shape[0]]).T
    DAD = {'DADsm_indices': DADsm_indices, 'DADsm_values': DADsm.data.astype(np.float32),
           'DADsp_indices': DADsp_indices, 'DADsp_values': DADsp.data.astype(np.float32), 'dense_shape': A.shape}
    sio.savemat(home_path + '/pre_built/{}_skeleton{}_{}.mat'.format(domain, time_input, num_adj), DAD)
    return DAD


def normalize_data(joint_data):
    center_x = np.array(joint_data[:, 0, 0])
    center_y = np.array(joint_data[:, 0, 1])
    for idx in range(len(joint_data)):
        joint_data[idx, :, 0] = (joint_data[idx, :, 0] / center_x[idx]) - (1 - normalized_point)
        joint_data[idx, :, 1] = (joint_data[idx, :, 1] / center_y[idx]) - (1 - normalized_point)
    return joint_data


def denormalize_data(joint_data, org):
    joint_data = joint_data.reshape(-1, time_input, 17, 2)
    org = org.reshape(-1, time_input, 17, 2)
    center_x = np.array(org[:, :, 0])
    center_y = np.array(org[:, :, 1])
    for batch_idx in range(len(joint_data)):
        for frame_idx in range(time_input):
            joint_data[batch_idx, frame_idx, :, 0] = (joint_data[batch_idx, frame_idx, :, 0] + (1 - normalized_point)) * \
                                                     center_x[batch_idx, frame_idx, 0]
            joint_data[batch_idx, frame_idx, :, 1] = (joint_data[batch_idx, frame_idx, :, 1] + (1 - normalized_point)) * \
                                                     center_y[batch_idx, frame_idx, 1]
    return joint_data


def load_pkl():
    if os.path.isfile(home_path + '/normalized_input/Normalized_data_{}_{}_{}.pkl'.format(normalized_point, time_input,
                                                                                          frame_interval)):
        joint_pkl_file = home_path + '/normalized_input/Normalized_data_{}_{}_{}.pkl'.format(normalized_point, time_input,
                                                                                       frame_interval)
        with open(joint_pkl_file, 'rb') as f:
            joint_data = pickle.load(f)

        bone_pkl_file = home_path + '/normalized_input/Normalized_bone_{}_{}.pkl'.format(time_input,
                                                                                       frame_interval)
        with open(bone_pkl_file, 'rb') as f:
            bone_data = pickle.load(f)

        org_pkl_file = home_path + '/normalized_input/Org_data_{}_{}_{}.pkl'.format(normalized_point, time_input,
                                                                                    frame_interval)
        with open(org_pkl_file, 'rb') as f:
            org_data = pickle.load(f)

    else:
        pkl_file = home_path + '/ntu60_hrnet.pkl'
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        joint_data = []
        org_data = []
        bone_data = []
        data = data['annotations']

        for num in range(len(data)):
            if len(data[num]['keypoint']) == 1:
                n_data = copy.deepcopy(data[num]['keypoint'][0])
                normalized = normalize_data(n_data)
                for num1 in range(0, len(normalized), frame_interval):
                    end = num1 + time_input
                    if end <= len(normalized):
                        joint_data.append(normalized[num1:end])
                        org_data.append(data[num]['keypoint'][0][num1:end])

        with open(home_path + '/normalized_input/Normalized_data_{}_{}_{}.pkl'.format(normalized_point, time_input,
                                                                                      frame_interval), 'wb') as f:
            pickle.dump(joint_data, f, pickle.HIGHEST_PROTOCOL)


        with open(home_path + '/normalized_input/Org_data_{}_{}_{}.pkl'.format(normalized_point, time_input,
                                                                               frame_interval), 'wb') as f:
            pickle.dump(org_data, f, pickle.HIGHEST_PROTOCOL)

    train_joint = copy.deepcopy(joint_data[0:-batch * 20])
    test_joint = copy.deepcopy(joint_data[-batch * 20:])
    train_bone = copy.deepcopy(bone_data[0:-batch * 20])
    test_bone = copy.deepcopy(bone_data[-batch * 20:])
    org_test = copy.deepcopy(org_data[-batch * 20:])
    return train_joint, test_joint, train_bone, test_bone, org_test


if __name__ == '__main__':
    train_lr = 1e-4
    finetune_lr = 1e-6
    weight_decay = 5e-4
    k = 20
    maximum_epoch = 300
    early_stopping = 20
    finetune_epoch = 50

    do_log = 1
    do_test = 100

    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    if os.path.isfile(home_path + '/pre_built/temporal_skeleton{}_{}.mat'.format(time_input, num_adj)):
        s_DAD = sio.loadmat(home_path + '/pre_built/temporal_skeleton{}_{}.mat'.format(time_input, num_adj))
        t_DAD = sio.loadmat(home_path + '/pre_built/spectral_skeleton{}_{}.mat'.format(time_input, num_adj))
    else:
        s_adj, t_adj = making_skeleton_adj()
        s_DAD = get_affinity_skeleton(s_adj, 'spectral')
        t_DAD = get_affinity_skeleton(t_adj, 'temporal')

    features_j, test_j, features_b, test_b, org_test = load_pkl()
    model_j = GALA.Model(s_DAD=s_DAD, t_DAD=t_DAD, name='GALA', batch_size=batch, trainable=True, time_input=time_input)
    model_b = GALA.Model(s_DAD=s_DAD, t_DAD=t_DAD, name='GALA', batch_size=batch, trainable=True, time_input=time_input)
    if args.load_model:
        model_j.built = True
        model_b.built = True
        model_j.load_weights('weights/t_s_1_adj1_30_15_10.h5', skip_mismatch=False, by_name=False,
                           options=None)
        model_b.load_weights('weights/t_s_1_adj1_30_15_10.h5', skip_mismatch=False, by_name=False,
                           options=None)
    init_step, init_loss_j, init_loss_b, validate, make_pkl, ACC, NMI, ARI = op_util.Optimizer(model_j, model_b,
                                                                                            [train_lr, finetune_lr])
    # training, train_loss, finetuning, validate, ACC, NMI, ARI

    summary_writer = tf.summary.create_file_writer(args.train_dir)
    with summary_writer.as_default():
        step = 0

        best_loss = 1e12
        stopping_step = 0

        train_time = time.time()
        # noised_inpute은 op_util.py에서 처리.
        if args.test:
            for epoch in range(maximum_epoch):
                temp = list(zip(features_j, features_b))
                random.shuffle(temp)
                features_j, features_b = zip(*temp)
                # res1 and res2 come out as tuples, and so must be converted to lists.
                features_j, features_b = list(features_j), list(features_b)
                tmp = len(features_j) // batch
                for num in range(len(features_j) // batch):
                    feature_j = np.array(features_j[num * batch:(num + 1) * batch]).astype(np.float32)
                    feature_j = feature_j.reshape(-1, 17 * time_input, 2)
                    feature_b = np.array(features_b[num * batch:(num + 1) * batch]).astype(np.float32)
                    feature_b = feature_b.reshape(-1, 17 * time_input, 2)
                    noised_j, noised_b, _, _ = make_noise(feature_j, feature_b)
                    init_step(feature_j, feature_b, noised_j, noised_b, weight_decay, k)
                step += 1
                model_j.save_weights('{}_J.h5'.format(args.save_model_name), overwrite=True, save_format=None, options=None)
                model_b.save_weights('{}_B.h5'.format(args.save_model_name), overwrite=True, save_format=None, options=None)
                # model.save('path_to_my_model', save_format='tf')

                if epoch % do_log == 0 or epoch == maximum_epoch - 1:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print(template.format(step, init_loss_j.result(), (time.time() - train_time) / do_log))
                    print(template.format(step, init_loss_b.result(), (time.time() - train_time) / do_log))
                    print()
                    train_time = time.time()
                    current_loss_j = init_loss_j.result()
                    current_loss_b = init_loss_b.result()
                    tf.summary.scalar('Initialization/train', current_loss_j, step=epoch + 1)
                    tf.summary.scalar('Initialization/train', current_loss_b, step=epoch + 1)
                    init_loss_j.reset_states()
                    init_loss_b.reset_states()

        else:
            # for num in range(len(test) // batch):
            #     tests = np.array(test[num * batch:(num + 1) * batch]).astype(np.float32)
            #     tests = tests.reshape(-1, 17 * time_input, 2)
            #     org_tests = np.array(org_test[num * batch:(num + 1) * batch]).astype(np.float32)
            #     org_tests = org_tests.reshape(-1, 17 * time_input, 2)
            #     validate(tests, org_tests, num)
            make_pkl(spt='xsub_val')