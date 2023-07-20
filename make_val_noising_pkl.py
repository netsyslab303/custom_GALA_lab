import os
import numpy as np
import pickle
import copy
import random

home_path = os.path.dirname(os.path.abspath(__file__))
time_input = 10
noise_ratio = 0.6


def make_score_nosing_pkl(spt):
    # ST-GCN++ 입력으로 사용 될 Noising pkl 파일 만들기 (score를 0으로 설정)
    noising_frame_num = []
    noising_joint_num = []
    pkl_file = home_path + '/ntu60_hrnet.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    if spt:
        split, data = data['split'], data['annotations']
        # WS
        identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        split = set(split[spt])
        data = [x for x in data if x[identifier] in split]

    noised_out = copy.deepcopy(data)
    for num in range(len(data)):
        tmp_frame = []
        tmp_joint = []
        for num_person in range(len(data[num]['keypoint_score'])):
            frame_num = len(data[num]['keypoint_score'][num_person])
            rnd_frame = random.sample(range(0, frame_num), int(frame_num * noise_ratio))
            rnd_joint = random.randint(0, 16)
            tmp_frame.append(rnd_frame)
            tmp_joint.append(rnd_joint)
            noised_out[num]['keypoint_score'][num_person][rnd_frame, rnd_joint] = 0
        noising_frame_num.append(tmp_frame)
        noising_joint_num.append(tmp_joint)
    with open(pkl_file, 'rb') as f:
        noised_data = pickle.load(f)
    noised_data['annotations'] = noised_out
    with open('Noising_score_{}_{}.pkl'.format(noise_ratio, spt), 'wb') as f:
        pickle.dump(noised_data, f, pickle.HIGHEST_PROTOCOL)
    with open("noising_frame_num_{}_{}.pkl".format(noise_ratio, spt), "wb") as f:
        pickle.dump(noising_frame_num, f, pickle.HIGHEST_PROTOCOL)
    with open("noising_joint_num_{}_{}.pkl".format(noise_ratio, spt), "wb") as f:
        pickle.dump(noising_joint_num, f, pickle.HIGHEST_PROTOCOL)


def make_keypoint_nosing_pkl(spt):
    # GALA 입력으로 사용 될 Noising pkl 파일 만들기 (keypoint를 0으로 설정)
    home_path = os.path.dirname(os.path.abspath(__file__))
    pkl_file = home_path + '/Noising_score_0.6_xsub_val.pkl'
    noising_frame_num = home_path + '/noising_frame_num_0.6_xsub_val.pkl'
    noising_joint_num = home_path + '/noising_joint_num_0.6_xsub_val.pkl'

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    with open(noising_frame_num, 'rb') as f:
        noising_frames = pickle.load(f)

    with open(noising_joint_num, 'rb') as f:
        noising_joints = pickle.load(f)

    if spt:
        split, data = data['split'], data['annotations']
        identifier = 'filename' if 'filen+ame' in data[0] else 'frame_dir'
        split = set(split[spt])
        annot = [x for x in data if x[identifier] in split]
    else:
        annot = None

    for num in range(len(annot)):
        for num_person in range(len(data[num]['keypoint_score'])):
            rnd_frame = noising_frames[num][num_person]
            rnd_joint = noising_joints[num][num_person]
            annot[num]['keypoint'][num_person][rnd_frame, rnd_joint, :] = 0

    with open(pkl_file, 'rb') as f:
        noised_data = pickle.load(f)
    noised_data['annotations'] = annot
    with open('Noising_keypoint_{}_{}.pkl'.format(noise_ratio, spt), 'wb') as f:
        pickle.dump(noised_data, f, pickle.HIGHEST_PROTOCOL)


make_score_nosing_pkl(spt='xsub_val')
make_keypoint_nosing_pkl(spt='xsub_val')
