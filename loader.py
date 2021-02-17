import numpy as np

def _parse_size(content):
    size = []
    for i in range(0, len(content), 4):
        size.append(int.from_bytes(content[i:i+4], "little"))
    return size

def _parse_voxels(content):
    n_voxels = int.from_bytes(content[:4], "little")
    voxels = []
    voxels_bytes = content[4:4+4*n_voxels]
    for i in range(0, len(voxels_bytes), 4):
        item = []
        for j in range(4):
            item.append(voxels_bytes[i+j])
        voxels.append(item)
    return voxels

def _parse_chunk_bytes(chunk_bytes):
    chunks = []
    p = 0
    while p < len(chunk_bytes):
        chunk_id = chunk_bytes[p:p+4].decode()
        N = int.from_bytes(chunk_bytes[p+4:p+8], "little")
        M = int.from_bytes(chunk_bytes[p+8:p+12], "little")
        content = chunk_bytes[p+12:p+12+N]
        children_bytes = chunk_bytes[p+12+N:p+12+N+M]
        p += 12+N+M
        children = None
        if len(children_bytes) > 0:
            children = _parse_chunk_bytes(children_bytes)
        if chunk_id == "MAIN":
            chunks.append({
                "chunk_id": chunk_id,
                "children": children
            })
        elif chunk_id == "SIZE":
            chunks.append({
                "chunk_id": chunk_id,
                "size": _parse_size(content)
            })
        elif chunk_id == "XYZI":
            chunks.append({
                "chunk_id": chunk_id,
                "voxels": _parse_voxels(content)
            })
    return chunks

def _trim_zero_planes(voxels):
    xl, xr = 0, 0
    for i in range(voxels.shape[0]):
        if (voxels[i, :, :] == 0).all(): xl = i
        else: break
    for i in range(voxels.shape[0] - 1, -1, -1):
        if (voxels[i, :, :] == 0).all(): xr = i
        else: break
    voxels = voxels[xl+1:xr, :, :]

    yl, yr = 0, 0
    for i in range(voxels.shape[1]):
        if (voxels[:, i, :] == 0).all(): yl = i
        else: break
    for i in range(voxels.shape[1] - 1, -1, -1):
        if (voxels[:, i, :] == 0).all(): yr = i
        else: break
    voxels = voxels[:, yl+1:yr, :]

    zl, zr = 0, 0
    for i in range(voxels.shape[2]):
        if (voxels[:, :, i] == 0).all(): zl = i
        else: break
    for i in range(voxels.shape[2] - 1, -1, -1):
        if (voxels[:, :, i] == 0).all(): zr = i
        else: break
    voxels = voxels[:, :, zl+1:zr]
    return voxels

def load_vox(model_path: str) -> np.ndarray:
    with open(model_path, "rb") as fin:
        _ = fin.read(8)
        chunk_bytes = fin.read()
    chunk = _parse_chunk_bytes(chunk_bytes)
    size = np.array(chunk[0]["children"][0]["size"])
    cloud = np.array(chunk[0]["children"][1]["voxels"])
    voxels = np.zeros(size)
    indices = cloud[:, :3]
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = cloud[:, 3]
    voxels = _trim_zero_planes(voxels)
    return voxels
