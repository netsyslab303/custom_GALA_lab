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

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="test", type=str)
parser.add_argument("--save_model_name", default="weight", type=str)
parser.add_argument("--load_model", action='store_true')
parser.add_argument("--test", action='store_false', help='for only_test')
args = parser.parse_args()
batch = 128
time_input = 30 #입력으로 사용되는 frame 수
frame_interval = 30 # Pkl 파일에서의 입력 간격 (Ex. frame :0 ~ 100 있는 PKL, time_input 30, frame_interval 15 => 0~30/ 15~45/ 30~60/ 45~75 ...
num_noise = 20 # 입력으로 사용되는 frame 중 noise 처리되는 frame 수



def make_noise(input_feature):
    noised_out = copy.deepcopy(input_feature)
    acc_rnd = []
    for batch_idx in range(len(input_feature)):
        # 동일 time_input 안 에서는 동일한 관절만 0으로 noising
        rnd_frame = random.sample(range(0,time_input),num_noise)
        rnd_joint = random.randint(0, 16)
        zero = [rnd_frame[i] * 17 + rnd_joint for i in range(len(rnd_frame))]
        noised_out[batch_idx][zero,:] = 0
            # rnd_idx = random.randint(0, 16)
            # radius = 0.05 * random.random()
            # theta = 2 * math.pi * random.random()
            # noised_out[idx, rnd_idx, 0] += radius * math.cos(theta)
            # noised_out[idx, rnd_idx, 1] += radius * math.sin(theta)
            # acc_rnd.append(rnd_idx)
    return noised_out.astype(np.float32), acc_rnd


def making_skeleton_adj():
    # 17*time_input 크기의 큰 인접행렬 생성
    adj_mat = np.zeros((17, 17))
    zero_mat = np.zeros((17, 17))
    idt_mat = np.eye(17)

    neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                     (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                     (1, 0), (3, 1), (2, 0), (4, 2)]
    for i, j in neighbor_link:
        adj_mat[i][j] = 1

    init_mat = [[zero_mat] * time_input for _ in range(time_input)]
    for i in range(time_input):
        init_mat[i][i] = copy.deepcopy(adj_mat)
        if i < time_input - 1:
            init_mat[i][i + 1] = idt_mat

    result = []
    for i in range(time_input):
        output = np.array(init_mat[i])
        reshaped_array = np.reshape(output, (17*time_input, 17)).T
        result.append(reshaped_array)
    adj_mat_out = np.concatenate(result)

    adj_csr = ss.csr_matrix(adj_mat_out)

    return adj_csr


def get_affinity_skeleton():
    A = making_skeleton_adj()
    A.toarray()
    A = A * np.exp(-1)
    A = A + A.T
    A = ss.csr_matrix.todense(A)
    eye = np.eye(A.shape[0])

    Asm = A + eye
    Dsm = 1 / np.sqrt(np.sum(Asm, -1))
    DADsm = ss.csr_matrix(np.multiply(np.multiply(Asm, Dsm).T, Dsm).reshape(-1))

    Asp = 2 * eye - A
    Dsp = 1 / np.sqrt(np.sum(2 * eye + A, -1))
    DADsp = ss.csr_matrix(np.multiply(np.multiply(Asp, Dsp).T, Dsp).reshape(-1))

    DADsm_indices = np.vstack([DADsm.indices // A.shape[0], DADsm.indices % A.shape[0]]).T
    DADsp_indices = np.vstack([DADsp.indices // A.shape[0], DADsp.indices % A.shape[0]]).T
    DAD = {'DADsm_indices': DADsm_indices, 'DADsm_values': DADsm.data.astype(np.float32),
           'DADsp_indices': DADsp_indices, 'DADsp_values': DADsp.data.astype(np.float32), 'dense_shape': A.shape}
    sio.savemat(home_path + '/pre_built/test_skeleton{}.mat'.format(time_input), DAD)
    return DAD


def normalize_data(joint_data):
    # x, y 좌표를 각각 정규화
    joint_data[:, :, 0] = (joint_data[:, :, 0] - (1920/2))/(1920/2)
    joint_data[:, :, 1] = (joint_data[:, :, 1] - (1080/2))/(1080/2)
    return joint_data


def denormalize_data(joint_data):
    # 원래 x,y 좌표로
    joint_data[:, :, 0] = joint_data[:, :, 0]*(1920/2)+(1920/2)
    joint_data[:, :, 1] = joint_data[:, :, 1]*(1080/2)+(1080/2)
    return joint_data


def load_pkl():
    pkl_file = home_path + '/ntu60_hrnet.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    joint_data = []
    data = data['annotations']

    for num in range(len(data)):
        for num_person in range(len(data[num]['keypoint'])):
            normalized = normalize_data(data[num]['keypoint'][num_person])
            for num1 in range(0, len(normalized), frame_interval):
                end = num1+time_input
                if end <= len(normalized):
                    joint_data.append(normalized[num1:end])

    train = copy.deepcopy(joint_data[0:-batch*20])
    test = copy.deepcopy(joint_data[-batch*20:])
    return train, test


if __name__ == '__main__':
    train_lr = 1e-4
    finetune_lr = 1e-6
    weight_decay = 5e-4
    k = 20
    maximum_epoch = 300
    early_stopping = 20
    finetune_epoch = 50

    do_log = 10
    do_test = 100

    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    if os.path.isfile(home_path + '/pre_built/test_skeleton{}.mat'.format(time_input)):
        DAD = sio.loadmat(home_path + '/pre_built/test_skeleton{}.mat'.format(time_input))
    else:
        DAD = get_affinity_skeleton()

    features, test = load_pkl()
    model = GALA.Model(DAD=DAD, name='GALA', batch_size=batch, trainable=True, time_input = time_input)
    if args.load_model:
        model.built = True
        model.load_weights('weight_0707.h5', skip_mismatch=False, by_name=False, options=None)
    init_step, init_loss, finetuning, validate, make_pkl, ACC, NMI, ARI = op_util.Optimizer(model,
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
                random.shuffle(features) #데이터셋을 매 epoch마다 shuffle (loss함수 치우치지 않도록)
                for num in range(len(features) // batch):
                    feature = np.array(features[num * batch:(num + 1) * batch]).astype(np.float32)
                    feature = feature.reshape(-1, 17*time_input, 2)
                    noised_input, _ = make_noise(feature)
                    init_step(feature, noised_input, weight_decay, k)
                step += 1
                model.save_weights('{}.h5'.format(args.save_model_name), overwrite=True, save_format=None, options=None)
                #model.save('path_to_my_model', save_format='tf')

                if epoch % do_log == 0 or epoch == maximum_epoch - 1:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print(template.format(step, init_loss.result(), (time.time() - train_time) / do_log))
                    train_time = time.time()
                    current_loss = init_loss.result()
                    tf.summary.scalar('Initialization/train', current_loss, step=epoch + 1)
                    init_loss.reset_states()

                if epoch % do_test == 0 or epoch == maximum_epoch - 1:
                    for num in range(len(test) // batch):
                        tests = np.array(test[num * batch:(num + 1) * batch]).astype(np.float32)
                        tests = tests.reshape(-1, 17*time_input, 2)
                        validate(tests, k)
                    tf.summary.scalar('Metrics/ACC', ACC.result(), step=epoch + 1)
                    tf.summary.scalar('Metrics/NMI', NMI.result(), step=epoch + 1)
                    tf.summary.scalar('Metrics/ARI', ARI.result(), step=epoch + 1)

                    template = 'Epoch: {0:3d}, NMI: {1:0.4f}, ARI.: {2:0.4f}'
                    print(template.format(epoch + 1, NMI.result(), ARI.result()))

                    NMI.reset_states()
                    ARI.reset_states()

                    params = {}
                    for v in model.variables:
                        params[v.name] = v.numpy()
                    sio.savemat(args.train_dir + '/trained_params.mat', params)

        else:
            make_pkl()
'''
        if finetune_epoch > 0:
            train_time = time.time()
            for epoch in range(finetune_epoch):
                for feature in test:
                    finetuning(feature, weight_decay, k)
                step += 1

            validate(test, k)
            tf.summary.scalar('Metrics/ACC', ACC.result(), step=maximum_epoch + epoch + 1)
            tf.summary.scalar('Metrics/NMI', NMI.result(), step=maximum_epoch + epoch + 1)
            tf.summary.scalar('Metrics/ARI', ARI.result(), step=maximum_epoch + epoch + 1)

            template = 'Epoch: {0:3d}, ACC: {1:0.4f}, NMI: {2:0.4f}, ARI.: {3:0.4f}'
            print(template.format(epoch + 1, ACC.result(), NMI.result(), ARI.result()))

            NMI.reset_states()
            ARI.reset_states()

        params = {}
        for v in model.variables:
            params[v.name] = v.numpy()
        sio.savemat(args.train_dir + '/tuned_params.mat', params)
'''
