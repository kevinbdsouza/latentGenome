import re
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from common import data_utils

'''
import numpy

>>>>>>> be71eac643ccc27ed80f0c5805289c6ce13cada6
npzPath = "/opt/data/latent/data/npz/subset_test_5_celltypes/test"

npzfiles = [f for f in listdir(npzPath) if isfile(join(npzPath, f))]

for file_name in npzfiles:
    npz_file_name = re.split(r"\.\s*", file_name)[0]

    new_path = npzPath + "/" + npz_file_name
    os.rename(file_name, new_path)

'''

'''
promoter_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/promoter"
enhancer_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/enhancer"

chr = 21

feature_matrix_promoter = pd.read_pickle(promoter_path + "/feat_chr_21_E116_promoter.pkl")
feature_matrix_enhancer = pd.read_pickle(enhancer_path + "/feat_chr_21_E116_enhancer.pkl")

print("done")
'''

dir_path = "/data2/latent/data/interpretation/phylogenetic_scores/"
chrom = 21
chrom_data = []

wig = dir_path + "chr21.phyloP100way.wigFix"
bigwig = dir_path + "hg19.100way.phyloP100way.bw"
bedgraph = dir_path + 'phyloP100way.chr{}.bedgraph'.format(chrom)
npz_path = dir_path + "chr21.phyloP100way.npz"
chrom_sizes = "/data2/latent/chrom.sizes"

os.system("/data2/latent/bigWigToBedGraph {} {} -chrom=chr{}".format(bigwig, bedgraph, chrom))

data = data_utils.bedgraph_to_dense(bedgraph, verbose=True)

data = data_utils.decimate_vector(data)

chrom_data.append(data)

np.savez(npz_path, chrom_data)

print("done")

'''
f = open(file_path)
for line in f:
    if line.startswith('fixedStep'):
        pass
    else:
        score = float(line.strip())
        s.append(score)
'''
