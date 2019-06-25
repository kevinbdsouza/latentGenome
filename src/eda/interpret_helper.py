from matplotlib import pyplot as plt
import numpy as np
import logging
import matplotlib.gridspec as gridspec
from downstream.pe_interactions import PeInteractions
from train_fns.test_gene import get_config
from common.log import setup_logging
from downstream.downstream_helper import DownstreamHelper
from downstream.run_downstream import DownstreamTasks
from Bio import SeqIO
import pandas as pd


class InterpretHelper:

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.chr_gc = 'chr' + str(chr)
        self.fasta_dir = "/data2/latent/data/dna/"
        self.gc_window = 25
        self.chr_len = cfg.chr_len[str(chr)]

    def get_gc_content(self):

        fasta_path = self.fasta_dir + self.chr_gc + ".fa"
        records = list(SeqIO.parse(fasta_path, "fasta"))

        gc_content = []
        for cut in range(int(len(records[0]) / self.gc_window)):
            new_seq = records[0].seq[cut * self.gc_window: (cut + 1) * self.gc_window].lower()

            gc_count = 0
            for pos in range(self.gc_window):

                if new_seq[pos] == "g" or new_seq[pos] == "c":
                    gc_count += 1

            gc_content.append(gc_count / self.gc_window)

        return np.array(gc_content)

    def create_mask(self):
        label_ar = np.zeros(self.chr_len)
        gene_ar = np.zeros(self.chr_len)
        mask_vec = np.ones(self.chr_len, bool)

        return mask_vec, label_ar, gene_ar

    def update_feature_matrix(self, feature_matrix, append_mat, mode):

        if mode == "gc":
            col = "gc_content"
        elif mode == "phylo":
            col = "p_score"

        non_zero = 1924795
        feature_matrix = feature_matrix.loc[:non_zero]

        append_mat = append_mat.reshape(-1, 1)
        append_mat = append_mat[:non_zero + 1, :]
        feature_matrix[col] = append_mat
        feature_matrix = feature_matrix.apply(pd.to_numeric)

        return feature_matrix
