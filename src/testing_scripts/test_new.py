import re
import os
from os import listdir
from os.path import isfile, join
import numpy as np

'''
npzPath = "/opt/data/latent/data/npz/subset_test_5_celltypes/test"

npzfiles = [f for f in listdir(npzPath) if isfile(join(npzPath, f))]

for file_name in npzfiles:
    npz_file_name = re.split(r"\.\s*", file_name)[0]

    new_path = npzPath + "/" + npz_file_name
    os.rename(file_name, new_path)

'''
