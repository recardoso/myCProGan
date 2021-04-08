import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc


from PIL import Image

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def convert_to_pil_image(image, drange=[0,255]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return Image.fromarray(image, format)

def save_image(image, filename, drange=[0,255], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)


if __name__ == "__main__":
    filename = 'image3.png'

    im=Image.open(filename)
    im.load()
    im = np.asarray(im, dtype=np.float32 )

    M = im.shape[0]//5
    N = im.shape[1]//5

    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

    final_tiles = []

    for t in tiles:
        t = t.transpose(2, 0, 1)
        final_tiles.append(t)

    print(final_tiles[1].shape)

    filename = 'cond4/2.png'

    for i in range(1,6):
        filename = 'cond4/'+str(i)+'.png'
        convert_to_pil_image(final_tiles[i-1], drange=[0,255]).save(filename)

    for i in range(1,5):
        filename = 'cond4/'+str(i*5+1)+'.png'
        convert_to_pil_image(final_tiles[i*5], drange=[0,255]).save(filename)

    #save_image(tiles[0], filename)