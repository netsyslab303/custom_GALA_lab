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


# def norm_euclidean_distance(input, output, acc):
#     dist = 0
#     for i in range(17):
#         dist += np.linalg.norm(input[i]-output[i])
#     return dist
#
# def euclidean_distance(input, output, acc):
#     dist = 0
#     for i in range(17):
#         input[i][0] *= 1920
#         input[i][1] *= 1080
#         output[i][0] *= 1920
#         output[i][1] *= 1080
#         dist += np.linalg.norm(input[i]-output[i])
#     return dist

def Optimizer(model, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        optimizer = tf.keras.optimizers.Adam(LR[0])
        optimizer_tune = tf.keras.optimizers.Adam(LR[1])

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        ACC = tf.keras.metrics.Mean(name='ACC')
        NMI = tf.keras.metrics.Mean(name='NMI')
        ARI = tf.keras.metrics.Mean(name='ARI')

    l = 5e1
    mu = 1.

    @tf.function
    def training(input, noised_input, weight_decay, k):
        with tf.GradientTape() as tape:
            generated = model(noised_input, training=True)
            loss = tf.reduce_sum(tf.square(input - generated)) / 2 / input.shape[0]
            total_loss = loss
            if weight_decay > 0.:
                total_loss += tf.add_n(
                    [tf.reduce_sum(tf.square(v)) * weight_decay / 2 for v in model.trainable_variables])
        # gradient 계
        gradients = tape.gradient(total_loss, model.trainable_variables)
        # 모델 업데이트
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)

    @tf.function
    def finetuning(input, weight_decay, k):
        with tf.GradientTape() as tape:
            generated = model(input, training=True)
            loss = tf.reduce_sum(tf.square(input - generated)) / 2 / input.shape[0]

            H = model.H
            HHT = tf.matmul(H, H, transpose_a=True)
            s, u, v = SVD.SVD(tf.expand_dims(mu * tf.eye(HHT.shape[0]) + l * HHT, 0), k)
            scc_loss = mu * l / 2 * tf.linalg.trace(
                tf.matmul(tf.squeeze(tf.matmul(v, u / tf.reshape(s, [1, 1, k]), transpose_b=True)), HHT))
            total_loss = loss + scc_loss

            if weight_decay > 0.:
                total_loss += tf.add_n([tf.reduce_sum(tf.square(v)) * weight_decay for v in model.trainable_variables])

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer_tune.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)

    def validate(input, labels, k):
        H = model(input, training=False)
        H = H.numpy()
        cv2.imwrite('test2/rec0.png',
                    (np.clip(np.hstack(list(np.hstack(list(H[:400].reshape(20, 20, 28, 28))))), 0, 1) * 255).astype(
                        np.uint8))
        cv2.imwrite('test2/rec1.png',
                    (np.clip(np.hstack(list(np.hstack(list(H[-400:].reshape(20, 20, 28, 28))))), 0, 1) * 255).astype(
                        np.uint8))
        cv2.imwrite('test2/ori.png',
                    (np.clip(np.hstack(list(np.hstack(list(input[:400].reshape(20, 20, 28, 28))))), 0, 1) * 255).astype(
                        np.uint8))
        cv2.imwrite('test2/ori2.png', (
                    np.clip(np.hstack(list(np.hstack(list(input[-400:].reshape(20, 20, 28, 28))))), 0, 1) * 255).astype(
            np.uint8))
        H = model.H
        latent = H.numpy()

        u, s, v = np.linalg.svd(mu * np.eye(latent.shape[1]) + l * np.matmul(latent.T, latent), full_matrices=False)
        H_inv = np.matmul(v[:k].T, (u[:, :k] / np.expand_dims(s[:k], 0)).T)
        A_latent = l * np.matmul(np.matmul(latent, H_inv), latent.T)
        A_latent = np.maximum(0, A_latent)
        # ss.save_npz('test2/test_full.npz', ss.csr_matrix(A_latent))
        print('Optimal A is computes')

        num_labels = np.max(labels) + 1

        clustering = SpectralClustering(n_clusters=num_labels, affinity='precomputed')
        prediction = clustering.fit(A_latent)

        results = prediction.labels_

        total_true = 0
        vote_box = {c: 0 for c in range(num_labels)}
        for i in range(num_labels):
            matched = labels[results == i]
            for m in matched:
                if vote_box.get(m) is not None:
                    vote_box[m] += 1

            num_true = 0
            cluster_id = 0
            for v in vote_box.keys():
                if vote_box[v] > num_true:
                    cluster_id = v
                    num_true = vote_box[v]

            total_true += num_true

            if num_true != 0:
                del vote_box[cluster_id]
            for v in vote_box.keys():
                vote_box[v] = 0

        acc = total_true / labels.shape[0]

        nmi_score = sklm.adjusted_mutual_info_score(labels.reshape(-1), results.reshape(-1))
        ari_score = sklm.adjusted_rand_score(labels.reshape(-1), results.reshape(-1))
        ACC.update_state(acc)
        NMI.update_state(nmi_score)
        ARI.update_state(ari_score)

    def validate2(input, org, batch):
        noised, noised_frame, noised_joint = train_ws_seperated_temporal.make_noise(input)
        H = model(noised, training=False)
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
        for num in range(len(annot)): #len(annot)
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
                input_1 = inputs[0:cut*time].reshape(-1, 17*time, 2)
                input_2 = inputs[-time:].reshape(-1, 17*time, 2)
                inputs = np.append(input_1,input_2, axis=0)
                org = copy.deepcopy(inputs)
                inputs = train_ws_seperated_temporal.normalize_data(inputs)
                H = train_ws_seperated_temporal.denormalize_data(np.array(model(inputs, training=False)),org).reshape(-1, 17, 2)
                output_1 = H[0:cut*time]
                output_2 = H[-res:]
                if res == 0:
                    output = output_1
                else:
                    output = np.append(output_1,output_2, axis=0)
                if len(data_annot[num]['keypoint'][num_person]) == len(output):
                    #data['annotations'][num]['keypoint'][num_person] = output
                    data['annotations'][num]['keypoint'][num_person][rnd_frame, rnd_joint, :] = output[rnd_frame, rnd_joint, :]

        with open('Denoising_point.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return training, train_loss, finetuning, validate2, make_pkl, ACC, NMI, ARI


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