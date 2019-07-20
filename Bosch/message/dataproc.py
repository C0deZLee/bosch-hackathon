from .eagle import *
from .vec2img import *

import os

images = []
label = []
import random
import pickle

data_path = "./data/"


def increase_data(d, l):
    images.append(d)
    label.append(l)
    with open(data_path+'images.pkl', 'wb') as fp:
        pickle.dump(images, fp)
    with open(data_path+'label.pkl', 'wb') as fp:
        pickle.dump(label, fp)


def get_data(root):
    global images, label

    if len(images) > 0:
        return images, label
    try:
        with open(data_path+'images.pkl', 'rb') as fp:
            images = pickle.load(fp)
        with open(data_path+'label.pkl', 'rb') as fp:
            label = pickle.load(fp)
        return images, label
    except:
        pass

    for i in range(10):
        for p, _, files in os.walk(data_path+"traindata/" + str(i)):
            for f in files:
                data = pd.read_csv(os.path.join(p, f))
                dt = data.get_values()
                c = csvDataToCoordinates(dt[:, 0], dt[:, 1], dt[:, 2], dt[:, 3] / (10 ** 9))
                c = np.array(c)
                # makePlot(c[:, 0], c[:, 1], c[:, 2])
                c = np.array(pts2flatten(c))
                im = pos2mnistlike(c[:, 0], c[:, 1])
                images.append(np.array(im))
                label.append(i)

    idx = [i for i in range(len(images))]
    random.shuffle(idx)
    images = np.array(images)[idx]
    label = np.array(label)[idx]

    with open(data_path+'images.pkl', 'wb') as fp:
        pickle.dump(images, fp)
    with open(data_path+'label.pkl', 'wb') as fp:
        pickle.dump(label, fp)

    return images, label
