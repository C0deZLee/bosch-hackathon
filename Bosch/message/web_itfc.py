from model import predict_one_record, save_one_record
from eagle import csvDataToCoordinates
from vec2img import pts2flatten, pts2image, pos2mnistlike


def predict(x, y, z, t):
    import numpy as np
    c = csvDataToCoordinates(x, y, z, np.array(t) / (10 ** 9))
    p2d = pts2flatten(c)
    img = pos2mnistlike(p2d[:, 0], p2d[:, 1])
    return predict_one_record(img)


def save_data(x, y, z, t, label):
    save_one_record([x, y, z, t], label)
