from loader import load_vox
from lego import legolize

voxels = load_vox("../models/VitaLemonTea.vox")
result = legolize(voxels)
print(result)
