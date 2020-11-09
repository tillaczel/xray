import numpy as np
from PIL import ImageOps


def augment(x, y, idx):
    x_l, y_l, idx_l = [], [], []
    for x_i, y_i, idx_i in zip(x, y, idx):
        x_l.append(x_i), y_l.append(y_i), idx_l.append(idx_i)

        x_a, y_a, idx_a = rotate(x_i, y_i, idx_i)
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = crop(x_i, y_i, idx_i)
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = flip(x_i, y_i, idx_i)
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = crop(*rotate(x_i, y_i, idx_i))
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = flip(*rotate(x_i, y_i, idx_i))
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = flip(*crop(x_i, y_i, idx_i))
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

        x_a, y_a, idx_a = flip(*crop(*rotate(x_i, y_i, idx_i)))
        x_l.append(x_a), y_l.append(y_a), idx_l.append(idx_a)

    return resize(x_l), np.array(y_l, dtype=float), idx_l, int(len(x_l)/len(x))


def flip(x, y, idx):
    return ImageOps.mirror(x), -y, f'{idx}_flip'


def crop(x, y, idx):
    x = ImageOps.expand(x.resize((192, 192)), border=32, fill='black')
    crop_cord = np.random.randint(0, 64)
    crop_cord = (crop_cord, 32, crop_cord + 192, 32 + 192)
    return x.crop(crop_cord, ), y, f'{idx}_crop'


def rotate(x, y, idx):
    width, height = x.size
    rotation = np.random.rand() * 20 - 10
    crop_cord = int(width / 2 * np.tan(abs(rotation) / 180 * np.pi))
    x = x.rotate(rotation, ).crop((crop_cord, crop_cord, width - crop_cord, width - crop_cord), )
    return x, y, f'{idx}_rotate'


def resize(x_l):
    x = [x_i.resize((64, 64)) for x_i in x_l]
    return x
