import copy
import math
import random
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.cluster import SpectralClustering
import scipy.sparse as ss
import sklearn.metrics as sklm
import cv2
from nets import SVD
import train_ws_seperated_temporal
import plot_output


def norm_euclidean_distance(input, output, acc):
    dist = np.linalg.norm(input[acc] - output[acc])
    return dist


def euclidean_distance(input, output, acc):
    dist = 0
    input[acc][0] *= 1920
    input[acc][1] *= 1080
    output[acc][0] *= 1920
    output[acc][1] *= 1080
    dist += np.linalg.norm(input[acc] - output[acc])
    return dist


def norm_euclidean_distance2(input, output, acc):
    dist = 0
    for i in range(17):
        dist += np.linalg.norm(input[i] - output[i])
    return dist


def euclidean_distance2(input, output, acc):
    dist = 0
    for i in range(17):
        input[i][0] *= 1920
        input[i][1] *= 1080
        output[i][0] *= 1920
        output[i][1] *= 1080
        dist += np.linalg.norm(input[i] - output[i])
    return dist


def joint_to_bone(joint):
    neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                     (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                     (1, 0), (3, 1), (2, 0), (4, 2)]
    joint = np.array(joint)
    bone = copy.deepcopy(joint)
    for i, j in neighbor_link:
        source = j
        target = i
        for k in range(joint.shape[1] // 17):
            bone[:, target + (k * 17), :] = joint[:, target + (k * 17), :] - joint[:, source + (k * 17), :]
    return bone


def bone_to_joint(bone):
    neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                     (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                     (1, 0), (3, 1), (2, 0), (4, 2)]
    bone = np.array(bone)
    joint = copy.deepcopy(bone)
    for i, j in neighbor_link:
        source = j
        target = i
        for k in range(joint.shape[1] // 17):
            joint[:, target + (k * 17), :] = joint[:, source + (k * 17), :] + bone[:, target + (k * 17), :]
    return joint


def Optimizer(model_j, model_b, LR):
    with tf.name_scope('Optimizer_w_Distillation_j'):
        j_optimizer = tf.keras.optimizers.Adam(LR[0])
        optimizer_tune = tf.keras.optimizers.Adam(LR[1])

        train_loss_j = tf.keras.metrics.Mean(name='train_loss_j')
        ACC = tf.keras.metrics.Mean(name='ACC')
        NMI = tf.keras.metrics.Mean(name='NMI')
        ARI = tf.keras.metrics.Mean(name='ARI')

    with tf.name_scope('Optimizer_w_Distillation_b'):
        b_optimizer = tf.keras.optimizers.Adam(LR[0])
        train_loss_b = tf.keras.metrics.Mean(name='train_loss_b')

    l = 5e1
    mu = 1.

    @tf.function
    def training(input_j, input_b, noised_joint, noised_bone, weight_decay, k):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            generated_j = model_j(noised_joint, training=True, info='joint')
            generated_b = model_b(noised_bone, training=True, info='bone')
            new_b = tf.py_function(joint_to_bone, inp=[generated_j], Tout=tf.float32)
            new_j = tf.py_function(bone_to_joint, inp=[generated_b], Tout=tf.float32)
            generated_j = (generated_j + new_j) / 2
            # generated_b = (generated_b + new_b) / 2
            loss_j = tf.reduce_sum(tf.square(input_j - generated_j)) / 2 / input_j.shape[0]
            loss_b = tf.reduce_sum(tf.square(input_b - generated_b)) / 2 / input_b.shape[0]
            total_loss_j = loss_j
            total_loss_b = loss_b
            if weight_decay > 0.:
                total_loss_j += tf.add_n(
                    [tf.reduce_sum(tf.square(v)) * weight_decay / 2 for v in model_j.trainable_variables])
                total_loss_b += tf.add_n(
                    [tf.reduce_sum(tf.square(v)) * weight_decay / 2 for v in model_b.trainable_variables])
        # gradient 계
        gradients_j = tape1.gradient(total_loss_j, model_j.trainable_variables)
        gradients_b = tape2.gradient(total_loss_b, model_b.trainable_variables)
        # 모델 업데이트
        j_optimizer.apply_gradients(zip(gradients_j, model_j.trainable_variables))
        b_optimizer.apply_gradients(zip(gradients_b, model_b.trainable_variables))
        train_loss_j.update_state(loss_j)
        train_loss_b.update_state(loss_b)

    @tf.function
    # def finetuning(input, weight_decay, k):
    #     with tf.GradientTape() as tape:
    #         generated = model(input, training=True)
    #         loss = tf.reduce_sum(tf.square(input - generated)) / 2 / input.shape[0]
    #
    #         H = model.H
    #         HHT = tf.matmul(H, H, transpose_a=True)
    #         s, u, v = SVD.SVD(tf.expand_dims(mu * tf.eye(HHT.shape[0]) + l * HHT, 0), k)
    #         scc_loss = mu * l / 2 * tf.linalg.trace(
    #             tf.matmul(tf.squeeze(tf.matmul(v, u / tf.reshape(s, [1, 1, k]), transpose_b=True)), HHT))
    #         total_loss = loss + scc_loss
    #
    #         if weight_decay > 0.:
    #             total_loss += tf.add_n([tf.reduce_sum(tf.square(v)) * weight_decay for v in model.trainable_variables])
    #
    #     gradients = tape.gradient(total_loss, model.trainable_variables)
    #     optimizer_tune.apply_gradients(zip(gradients, model.trainable_variables))
    #    train_loss.update_state(loss)

    def validate2(input, org, batch):
        noised, noised_frame, noised_joint = train_ws_seperated_temporal.make_noise(input)
        H = model_j(noised, training=False)  ## 수정필요
        output = np.array(H)
        output = train_ws_seperated_temporal.denormalize_data(output, org)
        plot_output.save_as_image(org, output, batch, noised_frame, noised_joint)
        # norm_dist = 0
        # dist = 0
        # for i in range(50):
        #     print("------------------")
        #     print(acc_rnd[i])
        #     print("Original")
        #     print(input[i])
        #     print("Noising")
        #     print(noised[i])
        #     print("Denoising")
        #     print(H[i].numpy())
        #     output = H[i].numpy()
        #     norm_dist += norm_euclidean_distance(input[i], output, acc_rnd[i])
        #     dist += euclidean_distance(input[i], output, acc_rnd[i])
        # print(norm_dist / 100)
        # print(dist / 100)
        # print(len(input))

    def make_pkl(spt):
        home_path = os.path.dirname(os.path.abspath(__file__))
        pkl_file = home_path + '/Noising_keypoint_0.3_xsub_val.pkl'
        noising_frame_num = home_path + '/noising_frame_num_0.3_xsub_val.pkl'
        noising_joint_num = home_path + '/noising_joint_num_0.3_xsub_val.pkl'
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        with open(noising_frame_num, 'rb') as f:
            noising_frames = pickle.load(f)
        with open(noising_joint_num, 'rb') as f:
            noising_joints = pickle.load(f)
        if spt:
            split, data_annot = data['split'], data['annotations']
            identifier = 'filename' if 'filen+ame' in data_annot[0] else 'frame_dir'
            split = set(split[spt])
            annot = [x for x in data_annot if x[identifier] in split]
        else:
            annot = data['annotations']
        for num in range(len(annot)):  # len(annot)
            if len(annot[num]['keypoint']) == 1:
                # for num_person in range(len(annot[num]['keypoint'])):
                # pkl 행동들의 frame수가 time_input으로 딱 나눠지지 않기 때문에 마지막 배열의 크기를 맞춰준다.
                num_person = 0
                rnd_frame = noising_frames[num][num_person]
                rnd_joint = noising_joints[num][num_person]
                inputs = annot[num]['keypoint'][num_person]
                time = train_ws_seperated_temporal.time_input
                cut = len(inputs) // time
                res = len(inputs) % time
                input_1 = inputs[0:cut * time].reshape(-1, 17 * time, 2)
                input_2 = inputs[-time:].reshape(-1, 17 * time, 2)
                inputs = np.append(input_1, input_2, axis=0)
                org = copy.deepcopy(inputs)
                inputs = train_ws_seperated_temporal.normalize_data(inputs)
                H = train_ws_seperated_temporal.denormalize_data(np.array(model_j(inputs, training=False)),
                                                                 org).reshape(
                    -1, 17, 2)  ## 수정필요
                output_1 = H[0:cut * time]
                output_2 = H[-res:]
                if res == 0:
                    output = output_1
                else:
                    output = np.append(output_1, output_2, axis=0)
                if len(data_annot[num]['keypoint'][num_person]) == len(output):
                    # data['annotations'][num]['keypoint'][num_person] = output
                    data['annotations'][num]['keypoint'][num_person][rnd_frame, rnd_joint, :] = output[rnd_frame,
                                                                                                rnd_joint, :]

        with open('Denoising_point.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return training, train_loss_j, train_loss_b, validate2, make_pkl, ACC, NMI, ARI

# home_path = os.path.dirname(os.path.abspath(__file__))
# pkl_file = home_path + '/Denoising.pkl'
# with open(pkl_file, 'rb') as f:
#     Denosing_data = pickle.load(f)
#
# pkl_file = home_path + '/Noising_keypoint_0.3_xsub_val.pkl'
# with open(pkl_file, 'rb') as f:
#     Nosing_data = pickle.load(f)
#
# print()
