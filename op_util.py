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

use_bone = train_ws_seperated_temporal.use_bone
use_polar = train_ws_seperated_temporal.use_polar
input_size = train_ws_seperated_temporal.input_size


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
    for k in range(joint.shape[1] // 17):
        bone[:, (k * 17), :] = 0
    for i, j in neighbor_link:
        source = j
        target = i
        for k in range(joint.shape[1] // 17):
            bone[:, target + (k * 17), :] = joint[:, target + (k * 17), :] - joint[:, source + (k * 17), :]
    if use_polar:
        x = bone[:, :, 0]
        y = bone[:, :, 1]
        polar_dis = np.sqrt(x ** 2 + y ** 2)
        polar_deg = np.arctan2(y, x)
        polar_deg[polar_deg < 0] += 2 * np.pi
        bone[:, :, 0] = polar_dis
        bone[:, :, 1] = polar_deg/(2 * np.pi)
    return bone


def bone_to_joint(bone):
    neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                     (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                     (1, 0), (3, 1), (2, 0), (4, 2)]
    bone = np.array(bone)
    tmp_bone = copy.deepcopy(bone)
    if use_polar:
        bone[:,:,0] = tmp_bone[:,:,0] * np.cos(tmp_bone[:,:,1]*(2 * np.pi))
        bone[:,:,1] = tmp_bone[:,:,0] * np.sin(tmp_bone[:,:,1]*(2 * np.pi))
    joint = copy.deepcopy(bone)
    # 코 (0,0)의 joint는 무조건 1,1로
    for k in range(joint.shape[1] // 17):
        joint[:, 0 + (k * 17), :] = 1
    for joints in range(1, 17):
        for i, j in neighbor_link:
            if i == joints:
                source = j
                target = i
                for k in range(joint.shape[1] // 17):
                    joint[:, target + (k * 17), :] = joint[:, source + (k * 17), :] + bone[:, target + (k * 17), :]
                break
    return joint


def Optimizer(model_j, model_b, LR):
    with tf.name_scope('Optimizer_w_Distillation_j'):
        j_optimizer = tf.keras.optimizers.Adam(LR[0])
        b_optimizer = tf.keras.optimizers.Adam(LR[0])
        optimizer_tune = tf.keras.optimizers.Adam(LR[1])

        train_loss_j = tf.keras.metrics.Mean(name='train_loss_j')
        train_loss_b = tf.keras.metrics.Mean(name='train_loss_b')
        ACC = tf.keras.metrics.Mean(name='ACC')
        NMI = tf.keras.metrics.Mean(name='NMI')
        ARI = tf.keras.metrics.Mean(name='ARI')

    l = 5e1
    mu = 1.

    @tf.function
    def training(input_j, input_b, noised_joint, noised_bone, weight_decay, k, dyna_adj):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            generated_j = model_j(noised_joint, training=True, dyna=dyna_adj)
            if use_bone:
                generated_b = model_b(noised_bone, training=True, dyna=dyna_adj)
                # new_b = tf.py_function(joint_to_bone, inp=[generated_j], Tout=tf.float32)
                # new_j = tf.py_function(bone_to_joint, inp=[generated_b], Tout=tf.float32)
                # generated_j = (generated_j + new_j) / 2
                # generated_b = (generated_b + new_b) / 2  #이 두줄로 서로의 연관성 조
                loss_b = tf.reduce_sum(tf.square(input_b - generated_b)) / 2 / input_b.shape[0]
                total_loss_b = loss_b
                total_loss_b += tf.add_n(
                    [tf.reduce_sum(tf.square(v)) * weight_decay / 2 for v in model_b.trainable_variables])
            loss_j = tf.reduce_sum(tf.square(input_j - generated_j)) / 2 / input_j.shape[0]
            total_loss_j = loss_j
            total_loss_j += tf.add_n(
                [tf.reduce_sum(tf.square(v)) * weight_decay / 2 for v in model_j.trainable_variables])
        # gradient 계
        gradients_j = tape1.gradient(total_loss_j, model_j.trainable_variables)
        # 모델 업데이트
        j_optimizer.apply_gradients(zip(gradients_j, model_j.trainable_variables))
        train_loss_j.update_state(loss_j)

        if use_bone:
            gradients_b = tape2.gradient(total_loss_b, model_b.trainable_variables)
            b_optimizer.apply_gradients(zip(gradients_b, model_b.trainable_variables))
            train_loss_b.update_state(loss_b)

    # @tf.function
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

    def validate2(noised_j, noised_b, noised_frame, noised_joint, org_j, batch, dyna_adj):
        generated_j = model_j(noised_j, training=False, dyna=dyna_adj)  ## 수정필요
        output = np.array(generated_j)
        output = train_ws_seperated_temporal.denormalize_data(output, org_j)
        plot_output.save_as_image(org_j, output, batch, noised_frame, noised_joint)
        if use_bone:
            generated_b = model_b(noised_b, training=False, dyna=dyna_adj)  ## 수정필요
            output = bone_to_joint(generated_b)
            output = train_ws_seperated_temporal.denormalize_data(output, org_j)
            plot_output.bone_save_as_image(output, batch, noised_frame, noised_joint)

    def make_pkl(spt):
        home_path = os.path.dirname(os.path.abspath(__file__))
        joint_pkl_file = home_path + '/Noising_keypoint_0.3_xsub_val.pkl'
        bone_pkl_file = home_path + '/Noising_bone_polar_0.3_xsub_val.pkl'
        noising_frame_num = home_path + '/noising_frame_num_0.3_xsub_val.pkl'
        noising_joint_num = home_path + '/noising_joint_num_0.3_xsub_val.pkl'
        with open(joint_pkl_file, 'rb') as f:
            joint_data = pickle.load(f)
        with open(bone_pkl_file, 'rb') as f:
            bone_data = pickle.load(f)
        with open(noising_frame_num, 'rb') as f:
            noising_frames = pickle.load(f)
        with open(noising_joint_num, 'rb') as f:
            noising_joints = pickle.load(f)
        if spt:
            split_j, data_annot_j = joint_data['split'], joint_data['annotations']
            split_b, data_annot_b = bone_data['split'], bone_data['annotations']
            identifier = 'filename' if 'filen+ame' in data_annot_j[0] else 'frame_dir'
            split = set(split_j[spt])
            annot_j = [x for x in data_annot_j if x[identifier] in split]
            annot_b = [x for x in data_annot_b if x[identifier] in split]
        else:
            annot_j = joint_data['annotations']
            annot_b = bone_data['annotations']
        for num in range(len(annot_j)):  # len(annot)
            if len(annot_j[num]['keypoint']) == 1:
                # for num_person in range(len(annot[num]['keypoint'])):
                # pkl 행동들의 frame수가 time_input으로 딱 나눠지지 않기 때문에 마지막 배열의 크기를 맞춰준다.
                num_person = 0
                rnd_frame = noising_frames[num][num_person]
                rnd_joint = noising_joints[num][num_person]
                # output_j = make_joint_or_bone_pkl(annot_j, annot_j, num, num_person, model_j, 'joint')
                # if len(data_annot_j[num]['keypoint'][num_person]) == len(output_j):
                #     # joint_data['annotations'][num]['keypoint'][num_person] = output_j
                #     joint_data['annotations'][num]['keypoint'][num_person][rnd_frame, rnd_joint, :] = output_j[rnd_frame, rnd_joint, :]

                if use_bone:
                    output_b = make_joint_or_bone_pkl(annot_b, annot_j, num, num_person, model_b, 'bone')
                    if len(data_annot_b[num]['keypoint'][num_person]) == len(output_b):
                        bone_data['annotations'][num]['keypoint'][num_person] = output_b
                        # joint_data['annotations'][num]['keypoint'][num_person][rnd_frame, rnd_joint, :] = output_b[
                        #                                                                                   rnd_frame,
                        #                                                                                   rnd_joint, :]
                '''
                inputs_j = annot_j[num]['keypoint'][num_person]
                score = annot_j[num]['keypoint_score'][num_person].reshape(inputs_j.shape[0], inputs_j.shape[1], 1)
                score[:, :, :] = 1
                score[np.array(rnd_frame), rnd_joint, :] = 0
                inputs_j = np.concatenate((inputs_j, score), axis=2)
                time = train_ws_seperated_temporal.time_input
                cut = len(inputs_j) // time
                res = len(inputs_j) % time
                input_1_j = inputs_j[0:cut * time].reshape(-1, 17 * time, 3)
                input_2_j = inputs_j[-time:].reshape(-1, 17 * time, 3)
                input_j = np.append(input_1_j, input_2_j, axis=0)
                org = copy.deepcopy(input_j[:,:,0:2])
                input_j = train_ws_seperated_temporal.normalize_data(input_j)
                if input_size == 3:
                    out_joint = train_ws_seperated_temporal.denormalize_data(np.array(model_j(input_j, training=False)),
                                                                     org).reshape(-1, 17, 2)  ## 수정필요
                else:
                    out_joint = train_ws_seperated_temporal.denormalize_data(np.array(model_j(input_j[:,:,0:2], training=False)),
                                                                     org).reshape(-1, 17, 2)  ## 수정필요

                output_1 = out_joint[0:cut * time]
                output_2 = out_joint[-res:]
                if res == 0:
                    output = output_1
                else:
                    output = np.append(output_1, output_2, axis=0)
                '''


        # with open('Denoising_point_j.pkl', 'wb') as f:
        #     pickle.dump(joint_data, f, pickle.HIGHEST_PROTOCOL)

        with open('Denoising_b_polar_only.pkl', 'wb') as f:
            pickle.dump(bone_data, f, pickle.HIGHEST_PROTOCOL)

    def make_joint_or_bone_pkl(inp, inp_j, num, num_person, model, inp_str):
        inputs = inp[num]['keypoint'][num_person]
        inputs_org = inp_j[num]['keypoint'][num_person]
        score = inp[num]['keypoint_score'][num_person].reshape(inputs.shape[0], inputs.shape[1], 1)
        inputs = np.concatenate((inputs, score), axis=2)

        time = train_ws_seperated_temporal.time_input
        cut = len(inputs) // time
        res = len(inputs) % time
        input_1 = inputs[0:cut * time].reshape(-1, 17 * time, 3)
        input_2 = inputs[-time:].reshape(-1, 17 * time, 3)
        org_inp_1 = inputs_org[0:cut * time].reshape(-1, 17 * time, 2)
        org_inp_2 = inputs_org[-time:].reshape(-1, 17 * time, 2)
        input = np.append(input_1, input_2, axis=0)
        org = np.append(org_inp_1, org_inp_2, axis=0)
        # input = train_ws_seperated_temporal.normalize_data(input) # Bone 일 때 확인필요

        if input_size == 3:
            model_out = model(input, training=False)
        else:
            model_out = model(input[:, :, 0:2], training=False)

        if inp_str == 'bone':

            model_out = bone_to_joint(model_out)

        out_joint = train_ws_seperated_temporal.denormalize_data(np.array(model_out),org).reshape(-1, 17, 2)
        output_1 = out_joint[0:cut * time]
        output_2 = out_joint[-res:]
        if res == 0:
            output = output_1
        else:
            output = np.append(output_1, output_2, axis=0)

        return output


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
