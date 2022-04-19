import numpy as np
from PIL import Image


def gauss_kernel(ksize=5, sigma=None, conv=False, dtype=np.float32):
    half = (ksize - 1) * 0.5
    if sigma is None:
        sigma = 0.3 * (half - 1) + 0.8
    x = np.arange(-half, half + 1)
    x = np.exp(- np.square(x / sigma) / 2)
    x = np.outer(x, x)
    x = x / x.sum()
    if conv:
        kernel = np.zeros((3, 3, ksize, ksize))
        for i in range(3):
            kernel[i, i] = x
    else:
        kernel = x
    return kernel.astype(dtype)


def pad_and_scale(img, lab=None, size=(416, 416), color=(127, 127, 127)):
    """

    Args:
        img:

    Returns:

    """
    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=color)
            padded_img.paste(img, (int(padding), 0))
            if lab is not None:
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=color)
            padded_img.paste(img, (0, int(padding)))
            if lab is not None:
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
    padded_img = padded_img.resize((size[0], size[1]))
    if lab is None:
        return padded_img
    else:
        return padded_img, lab


def random_crop(cloth, crop_size, pos=None, crop_type=None):
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size is 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]
        patch = cloth[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h


def random_stick(inputs, patch, stick_size=None, mode='replace', pos=None):
    if stick_size is None:
        stick_size = patch.shape[2:4]
    w = inputs.shape[2]
    h = inputs.shape[3]
    d_w = w - stick_size[0]
    d_h = h - stick_size[1]
    if pos is None:
        r_w = np.random.randint(d_w + 1)
        r_h = np.random.randint(d_h + 1)
    elif pos == 'center':
        r_w, r_h = (np.array(inputs.shape[2:]) - np.array(stick_size)) // 2
    else:
        r_w = pos[0]
        r_h = pos[1]

    patch_stick = inputs.new_zeros(inputs.shape)
    patch_resized = patch
    patch_stick[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = patch_resized

    assert mode in ['add', 'replace']

    if mode == 'add':
        inputs_stick = (inputs + patch_stick).clamp(0, 1)
    #         return inputs_stick

    elif mode == 'replace':
        mask = inputs.new_zeros(inputs.shape)
        mask[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = 1
        inputs_stick = mask * patch_stick + (1 - mask) * inputs
        inputs_stick = inputs_stick.clamp(0, 1)
    else:
        inputs_stick = None

    return inputs_stick, r_w, r_h


def TVLoss(patch):

    t1 = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().sum()
    t2 = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().sum()

    tv = t1 + t2

    return tv / patch.numel()

