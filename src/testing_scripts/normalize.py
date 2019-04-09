import re
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math

npzPath = "/opt/data/latent/data/npz/all_npz"
z_norm_path = "/opt/data/latent/data/npz/all_npz_arc_sinh"

npzfiles = [f for f in listdir(npzPath) if isfile(join(npzPath, f))]

mean_vec = []
std_vec = []

for i, f in enumerate(npzfiles):
    npz_file_name = re.split(r"\.\s*", f)[0]
    dat = np.load(npzPath + "/" + f)["arr_0"]

    arc_sin_dat = np.log(dat + np.sqrt(1 + np.power(dat, 2)))
    # u = np.mean(dat)
    # v = np.std(dat)
    # z_norm_dat = (dat - u) / v
    np.savez(z_norm_path + "/" + npz_file_name, arc_sin_dat)

print("done")
