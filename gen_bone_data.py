import copy
import os
import pickle
import argparse
import train_ws_seperated_temporal

parser = argparse.ArgumentParser()
parser.add_argument("--time_input", default=30, type=str)
parser.add_argument("--frame_interval", default=100, type=int)
args = parser.parse_args()

neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                 (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                 (1, 0), (3, 1), (2, 0), (4, 2)]

home_path = os.path.dirname(os.path.abspath(__file__))
#pkl_file = home_path + '/normalized_input/Normalized_bone_30_30.pkl'
pkl_file = home_path + '/ntu60_hrnet.pkl'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

joint_data = []
org_data = []
time_input = args.time_input
frame_interval = args.frame_interval

for data_len in range(len(data['annotations'])):
    num_person = len(data['annotations'][data_len]['keypoint'])
    source = 0
    if num_person == 1:
        key_points = train_ws_seperated_temporal.normalize_data(data['annotations'][data_len]['keypoint'][0])
        bone = copy.deepcopy(key_points)
        bone[:, 0, :] = key_points[:, source, :] - key_points[:, 0, :]
        for i, j in neighbor_link:
            source = j
            target = i
            bone[:, target, :] = key_points[:, target, :] - key_points[:, source, :]
        for num1 in range(0, len(bone), frame_interval):
            end = num1 + time_input
            if end <= len(bone):
                joint_data.append(bone[num1:end])

        #data['annotations'][data_len]['keypoint'][0] = copy.deepcopy(bone)
with open(home_path + '/normalized_input/Normalized_bone_{}_{}.pkl'.format(time_input,frame_interval), 'wb') as f:
    pickle.dump(joint_data, f, pickle.HIGHEST_PROTOCOL)