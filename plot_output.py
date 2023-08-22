import math
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
from numpy import linalg as LA

path = os.getcwd()
input = np.load('input.npy')
ouput = np.load('output.npy')
neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                 (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                 (1, 0), (3, 1), (2, 0), (4, 2)]


def save_as_image(input, output, batch, noised_frame, noised_joint):
    inp = input.reshape([len(input), -1, 17, 2])
    oup = output.reshape([len(output), -1, 17, 2])
    batch_dir = path + '/images/batch_{}'.format(batch)
    if os.path.exists(batch_dir):
        pass
    else:
        os.makedirs(batch_dir)
    np.save(batch_dir + "/noised_frame", noised_frame)
    np.save(batch_dir + "/noised_joint", noised_joint)
    for i in range(len(inp)):
        image_dir = path + '/images/batch_{}/{}'.format(batch, i)
        if os.path.exists(image_dir):
            pass
        else:
            os.makedirs(image_dir)
        np.save(image_dir + "/input", inp)
        np.save(image_dir + "/output", oup)
        np.save(image_dir + "/noised_frame_{}_{}_{}".format(noised_frame[i][0], noised_frame[i][1], noised_frame[i][2]),
                noised_frame[i])
        for j in (noised_frame[i]):
            x = inp[i][j][:, 0]
            y = inp[i][j][:, 1]
            plt.xlim(-1920, 0)
            plt.ylim(-1080, 0)
            plt.scatter(-x, -y, s=10, zorder=0)
            if j in noised_frame[i][:]:
                joint = noised_joint[i]
                plt.scatter(-inp[i][j][joint][0], -inp[i][j][joint][1], s=20, edgecolors='black', marker='X', zorder=1)
            for a in neighbor_link:
                x1, x2 = a
                plt.plot([-x[x1], -x[x2]], [-y[x1], -y[x2]], 'b-', zorder=0, alpha=0.5)
            plt.savefig(os.path.join(image_dir, 'org_{}.pdf'.format(j)))
            plt.close()

            plt.xlim(-1920, 0)
            plt.ylim(-1080, 0)
            o = oup[i][j][:, 0]
            p = oup[i][j][:, 1]
            plt.scatter(-o, -p, s=10, zorder=0)
            if j in noised_frame[i][:]:
                joint = noised_joint[i]
                plt.scatter(-oup[i][j][joint][0], -oup[i][j][joint][1], s=20, edgecolors='black', marker='*', zorder=1)
            for a in neighbor_link:
                x1, x2 = a
                plt.plot([-o[x1], -o[x2]], [-p[x1], -p[x2]], 'b-', zorder=0, alpha=0.5)
            plt.savefig(os.path.join(image_dir, 'out_{}_j.pdf'.format(j)))
            plt.close()
            # plt.show()


def bone_save_as_image(output, batch, noised_frame, noised_joint):
    oup = output.reshape([len(output), -1, 17, 2])
    batch_dir = path + '/images/batch_{}'.format(batch)
    for i in range(len(oup)):
        image_dir = path + '/images/batch_{}/{}'.format(batch, i)
        for j in (noised_frame[i]):
            plt.xlim(-1920, 0)
            plt.ylim(-1080, 0)
            o = oup[i][j][:, 0]
            p = oup[i][j][:, 1]
            plt.scatter(-o, -p, s=10, zorder=0)
            if j in noised_frame[i][:]:
                joint = noised_joint[i]
                plt.scatter(-oup[i][j][joint][0], -oup[i][j][joint][1], s=20, edgecolors='black', marker='*', zorder=1)
            for a in neighbor_link:
                x1, x2 = a
                plt.plot([-o[x1], -o[x2]], [-p[x1], -p[x2]], 'b-', zorder=0, alpha=0.5)
            plt.savefig(os.path.join(image_dir, 'out_{}_b.pdf'.format(j)))
            plt.close()


def image_to_gif():
    image_dirs = path + '/images/'
    for image_dir in os.listdir(image_dirs):
        dir_list = os.path.join(image_dirs, image_dir)
        image_list = os.listdir(dir_list)
        image_list.sort()
        images = []
        for filename in image_list:
            images.append(imageio.imread(os.path.join(dir_list, filename)))
        imageio.mimsave('./gif/{}.gif'.format(image_dir), images, 'GIF', duration=0.5, loop=1)
