import numpy as np
import re
import pandas as pd
import gzip
import os
from os import listdir
from os.path import join, isfile

rna_seq_path = "/opt/data/latent/data/downstream/RNA-seq"
pe_int_path = "/opt/data/latent/data/downstream/PE-interactions"

# rna-seq data
gene_info = pd.read_csv(rna_seq_path + '/Ensembl_v65.Gencode_v10.ENSG.gene_info', sep="\s+", header=None)

pc_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.pc.gz", compression='gzip', header=0, sep="\s+",
                      error_bad_lines=False)
nc_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.nc.gz", compression='gzip', header=0, sep="\s+",
                      error_bad_lines=False)
rb_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.rb.gz", compression='gzip', header=0, sep="\s+",
                      error_bad_lines=False)


# pe-interactions
nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end',
                 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window',
                 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

training_df = pd.read_hdf(pe_int_path + '/training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
labels = training_df['label']

pe_pairs = pd.read_csv(pe_int_path + '/pairs.csv', sep=",")

print("done")