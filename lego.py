from typing import List, Tuple
from itertools import combinations
from scipy.signal import convolve2d

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

def get_growth_map(mask: np.ndarray) -> List[Tuple]:
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

def grow(plane: np.ndarray) -> List[Tuple]:
    mask = np.zeros_like(plane)
    x, y = plane.shape
    gmap = get_growth_map(mask)
    points = np.array(np.where(gmap != -1)).T.tolist()
    block_idx = 1
    while len(points) > 0:
        i, j = random.choice(points)
        allow = gmap[i, j]
        bxs = [2, 1]
        bys = [10, 8, 6, 4, 3, 2, 1]
        random.shuffle(bxs)
        random.shuffle(bys)
        for bx in bxs:
            break_flag = False
            if allow & 1:
                bx = -bx
            for by in bys:
                if allow & 4:
                    by = -by
                to_place = mask[i:i+bx:[-1,1][bx>0], j:j+by:[-1,1][by>0]]
                if (to_place == 0).all() and (0 <= i+bx <= x) and (0 <= j+by <= y):
                    mask[i:i+bx:[-1,1][bx>0], j:j+by:[-1,1][by>0]] = block_idx
                    block_idx += 1
                    break_flag = True
                    break
                to_place = mask[i:i+by:[-1,1][by>0], j:j+bx:[-1,1][bx>0]]
                if (to_place == 0).all() and (0 <= i+by <= x) and (0 <= j+bx <= y):
                    mask[i:i+by:[-1,1][by>0], j:j+bx:[-1,1][bx>0]] = block_idx
                    block_idx += 1
                    break_flag = True
                    break
            if break_flag:
                break
        gmap = get_growth_map(mask)
        points = np.array(np.where(gmap != -1)).T.tolist()
    return mask

if __name__ == "__main__":
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    growth_map = get_growth_map(mask)
    print(growth_map)
    mask = grow(mask)
    print(mask)