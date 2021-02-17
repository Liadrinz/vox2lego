from scipy.signal import convolve2d

import tqdm
import random
import numpy as np

KERNELS = [
    np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]]),
    np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0]]),
    np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]]),
    np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]]),
]

def self_similarity(array: np.ndarray) -> int:
    array = array.reshape([-1,])
    length = array.shape[0]
    array = np.tile(array[np.newaxis, ...], [length, 1])
    array_a = np.reshape(array, [-1,])
    array_b = np.reshape(array.T, [-1,])
    return np.sum((array_a == array_b).astype(np.int)) / (length * length)

def get_growth_map(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.bool)
    gmaps = []
    forbids = []
    for i, kernel in enumerate(KERNELS):
        gmap = (convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=1)//2).astype(np.bool)
        gmap &= (1 - mask).astype(np.bool)
        forbids.append(gmap)
        gmap = gmap.astype(np.int) * 2 ** i
        gmaps.append(gmap)
    gmaps = np.sum(gmaps, axis=0)
    l_allow = (1 - forbids[0]) * (1 - forbids[3]) * 1
    r_allow = (1 - forbids[1]) * (1 - forbids[2]) * 2
    u_allow = (1 - forbids[0]) * (1 - forbids[1]) * 4
    d_allow = (1 - forbids[3]) * (1 - forbids[2]) * 8
    amaps = l_allow + r_allow + u_allow + d_allow
    gmask = gmaps.astype(np.bool).astype(np.int)
    return amaps * gmask - (1 - gmask)

def legolize(vox: np.ndarray) -> np.ndarray:
    result = []
    vox = np.transpose(vox, [2, 0, 1])
    with tqdm.tqdm(vox) as pbar:
        for plane in vox:
            spmap = np.zeros_like(plane)
            x, y = plane.shape
            is_blank = (plane == 0).astype(np.int)
            spmap = np.where(is_blank, -1, spmap)
            gmap = get_growth_map(spmap)
            points = np.array(np.where(gmap != -1)).T.tolist()
            block_idx = np.max(spmap) + 1
            while len(points) > 0:
                i, j = random.choice(points)
                allow = gmap[i, j]
                bxs = [2, 1]
                bys = [10, 8, 6, 4, 3, 2, 1]
                for bx in bxs:
                    break_flag = False
                    if allow & 1:
                        bx = -bx
                    for by in bys:
                        if allow & 4:
                            by = -by
                        sx = [-1,1][bx>0]
                        sy = [-1,1][by>0]
                        to_place = spmap[i:i+bx:sx, j:j+by:sy]
                        to_place_color = plane[i:i+bx:sx, j:j+by:sy]
                        if (to_place == 0).all() and (0 <= i+bx <= x) and (0 <= j+by <= y) and self_similarity(to_place_color) == 1.0:
                            spmap[i:i+bx:sx, j:j+by:sy] = block_idx
                            block_idx += 1
                            break_flag = True
                            break
                        to_place = spmap[i:i+by:sy, j:j+bx:sx]
                        if (to_place == 0).all() and (0 <= i+by <= x) and (0 <= j+bx <= y) and self_similarity(to_place_color) == 1.0:
                            spmap[i:i+by:sy, j:j+bx:sx] = block_idx
                            block_idx += 1
                            break_flag = True
                            break
                    if break_flag:
                        break
                gmap = get_growth_map(spmap)
                points = np.array(np.where(gmap != -1)).T.tolist()
            result.append(spmap)
            pbar.update(1)
    return np.array(result)

if __name__ == "__main__":
    mask = np.zeros((6, 6, 6))
    spmap = legolize(mask)
