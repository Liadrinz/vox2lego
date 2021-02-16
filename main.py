from vox_loader import load_vox
from lego import grow

voxels = load_vox("../models/VitaLemonTea.vox")
result = grow(voxels[:, :, 0])
print(result)
