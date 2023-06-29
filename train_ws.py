import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import os, time, argparse
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import op_util
from nets import GALA

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="test", type=str)
args = parser.parse_args()
batch = 256
num_data = 256*250


def making_skeleton_adj():
    adj_mat = np.array([[0] * 17] * 17)
    neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                     (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                     (1, 0), (3, 1), (2, 0), (4, 2)]
    for i, j in (neighbor_link):
        adj_mat[i][j] = 1

    adj_csr = ss.csr_matrix(adj_mat)

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
    sio.savemat(home_path + '/pre_built/test_skeleton.mat', DAD)
    return DAD


def normalize_data(joint_data):
    # x, y 좌표를 각각 정규화
    normalized_x = (joint_data[:, ::2]) / 1920
    normalized_y = (joint_data[:, 1::2]) / 1080

    normalized_data = np.concatenate((normalized_x, normalized_y), axis=1, dtype=np.float32)
    return normalized_data


def load_pkl():
    pkl_file = home_path + '/ntu60_hrnet.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    joint_data = []
    data = data['annotations']
    count = 0

    nn = num_data
    for num in range(len(data)):
        for num1 in range(0, len(data[num]['keypoint'][0]), 30):
            joint_data.append(data[num]['keypoint'][0][num1])
            count += 1
            if count == nn:
                break

    train = []
    test = []
    for t in range(nn):
        normalized_data = normalize_data(joint_data[t])

        if t <= nn - (256 * 2):
            train.append(normalized_data)
        else:
            test.append(normalized_data)
    return train, test


if __name__ == '__main__':
    train_lr = 1e-4
    finetune_lr = 1e-6
    weight_decay = 5e-4
    k = 20
    maximum_epoch = 1000
    early_stopping = 20
    finetune_epoch = 50

    do_log = 10
    do_test = 100

    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    if os.path.isfile( home_path+'/pre_built/test_skeleton.mat'):
        DAD = sio.loadmat(home_path + '/pre_built/test_skeleton.mat')
    else:
        DAD = get_affinity_skeleton()

    features, test = load_pkl()
    model = GALA.Model(DAD=DAD, name='GALA', batch_size=batch, trainable=True)

    init_step, init_loss, finetuning, validate, ACC, NMI, ARI = op_util.Optimizer(model, [train_lr, finetune_lr])
    # training, train_loss, finetuning, validate, ACC, NMI, ARI

    summary_writer = tf.summary.create_file_writer(args.train_dir)
    with summary_writer.as_default():
        step = 0

        best_loss = 1e12
        stopping_step = 0

        train_time = time.time()
        for epoch in range(maximum_epoch):
            for num in range(len(features) // batch):
                feature = np.array(features[num * batch:(num + 1) * batch])
                feature = feature.reshape(-1, 17, 2)
                init_step(feature, weight_decay, k)
            step += 1

            if epoch % do_log == 0 or epoch == maximum_epoch - 1:
                template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                print(template.format(step, init_loss.result(), (time.time() - train_time) / do_log))
                train_time = time.time()
                current_loss = init_loss.result()
                tf.summary.scalar('Initialization/train', current_loss, step=epoch + 1)
                init_loss.reset_states()

            if epoch % do_test == 0 or epoch == maximum_epoch - 1:
                for num in range(len(test) // batch):
                    tests = np.array(test[num * batch:(num + 1) * batch])
                    tests = tests.reshape(-1, 17, 2)
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
        model.save_weights('weight.h5')
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