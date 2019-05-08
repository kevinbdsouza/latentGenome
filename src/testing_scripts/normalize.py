import re
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math

npzPath = "/data2/latent/data/npz/ch20_arc_sinh"
z_norm_path = "/data2/latent/data/npz/ch_20_arc_sinh_znorm"

npzfiles = [f for f in listdir(npzPath) if isfile(join(npzPath, f))]

mean_vec = []
std_vec = []

for i, f in enumerate(npzfiles):
    npz_file_name = re.split(r"\.\s*", f)[0]
    dat = np.load(npzPath + "/" + f)["arr_0"]

    # arc_sin_dat = np.log(dat + np.sqrt(1 + np.power(dat, 2)))

    u = np.mean(dat)
    v = np.std(dat)
    z_norm_dat = (dat - u) / v

    np.savez(z_norm_path + "/" + npz_file_name, z_norm_dat)

print("done")
