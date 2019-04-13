import numpy as np

path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/all_ca_arc_sinh/"

rna = np.load(path + "map_dict_rnaseq.npy").item()

pe = np.load(path + "map_dict_pe.npy").item()

fire = np.load(path + "map_dict_fire.npy").item()

tad = np.load(path + "map_dict_tad.npy").item()

print("done")