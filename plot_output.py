import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg

path = os.getcwd()
input = np.load('input.npy')
ouput = np.load('output.npy')
neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                 (9, 7), (7, 5), (10, 8), (8, 6), (5, 3), (6, 4),
                 (1, 0), (3, 1), (2, 0), (4, 2)]


def save_as_image(input, output):
    inp = input.reshape([len(input), -1, 17, 2])
    oup = ouput.reshape([len(output), -1, 17, 2])
    for i in range(len(inp)):
        image_dir = path + '/images/{}'.format(i)
        if os.path.exists(image_dir):
            pass
        else:
            os.makedirs(image_dir)
        for j in range(len(inp[i])):
            x = inp[i][j][:, 0]
            y = inp[i][j][:, 1]
            o = oup[i][j][:, 0]
            p = oup[i][j][:, 1]
            plt.scatter(1 - x, 1 - y, s=10)
            plt.scatter(1 - o, 1 - p, s=10)
            for a in neighbor_link:
                x1, x2 = a
                plt.plot([1 - x[x1], 1 - x[x2]], [1 - y[x1], 1 - y[x2]], 'b-')
                plt.plot([1 - o[x1], 1 - o[x2]], [1 - p[x1], 1 - p[x2]], 'r-')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            # plt.show()
            plt.savefig(os.path.join(image_dir, '{}.png'.format(j)))
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
        imageio.mimsave('./images/{}.gif'.format(image_dir), images, 'GIF', duration=0.5, loop=1)


image_to_gif()
