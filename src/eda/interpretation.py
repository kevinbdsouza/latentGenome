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
from eda.interpret_helper import InterpretHelper
import pandas as pd


class Interpretation:

    def __init__(self, cfg, dir, chr, mode):
        self.data_dir = "/data2/latent/data/"
        self.pe_int_path = self.data_dir + "downstream/PE-interactions"
        self.phylo_path = "/data2/latent/data/interpretation/phylogenetic_scores/"
        self.pe_cell_names = ['E123', 'E117', 'E116', 'E017']
        self.chr_pe = 'chr' + str(chr)
        self.chr_gc = 'chr' + str(chr)
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode=mode)
        self.downstream_ob = DownstreamTasks(cfg, dir, chr, mode)
        self.saved_model_dir = dir
        self.feat_mat_pe = self.saved_model_dir + "feat_chr_" + str(chr)
        self.feat_mat_whole = self.saved_model_dir + "feat_chr_" + str(chr)
        self.feat_mat_gc = self.saved_model_dir + "feat_gc_chr_" + str(chr)
        self.feat_mat_phylo = self.saved_model_dir + "feat_phylo_chr_" + str(chr)
        self.run_features_pe = True
        self.run_features_gc = True
        self.interpret_helper_ob = InterpretHelper(cfg, chr)

    @staticmethod
    def update_config(model_path, config_base, result_base):
        cfg = get_config(model_path, config_base, result_base)
        pd_col = list(np.arange(cfg.hidden_size_encoder))
        pd_col.append('target')
        pd_col.append('gene_id')
        cfg = cfg._replace(downstream_df_columns=pd_col)

        return cfg

    def get_pe_df(self, cfg):
        pe_df = None

        pe_ob = PeInteractions()
        pe_ob.get_pe_data(intp_ob.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(intp_ob.chr_pe)

        element_list = ["promoter", "enhancer"]
        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]

            for e in element_list:
                pe_window_labels = pe_data_chr_cell.filter([e + '_start', e + '_end', 'label'], axis=1)
                pe_window_labels.rename(columns={e + '_start': 'start', e + '_end': 'end', 'label': 'target'},
                                        inplace=True)
                pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

                mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(pe_window_labels)

                feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                              self.run_features_pe,
                                                                              self.feat_mat_pe + "_" + cell + "_" + e + ".pkl",
                                                                              self.downstream_ob.downstream_main,
                                                                              self.chr_pe)

                feature_matrix = self.downstream_helper_ob.get_window_features(feature_matrix)

                feature_matrix.to_pickle(
                    self.saved_model_dir + e + "/feat_chr_" + str(chr) + "_" + cell + "_" + e + ".pkl")

                logging.info(
                    "Pickle created for - chr : {} - cell : {} - element : {}".format(str(self.chr_pe), cell, e))

        return pe_df

    def get_gc_df(self, cfg):

        gc_df = None

        '''
        gc_content = self.interpret_helper_ob.get_gc_content()
        np.save(self.saved_model_dir + "gc_content.npy", gc_content)
        mask_vector, label_ar, gene_ar = self.interpret_helper_ob.create_mask()
        feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                      self.run_features_gc,
                                                                      self.feat_mat_whole + ".pkl",
                                                                      self.downstream_ob.downstream_main,
                                                                      self.chr_gc)

        '''

        gc_content = np.load(self.saved_model_dir + "gc_content.npy")
        feature_matrix = pd.read_pickle(self.feat_mat_whole + ".pkl")

        feature_matrix = self.interpret_helper_ob.update_feature_matrix(feature_matrix, gc_content, mode="gc")

        feature_matrix.to_pickle(self.feat_mat_gc + ".pkl")

        logging.info("Pickle created for - chr : {}".format(str(self.chr_gc)))

        return gc_df, gc_content

    def get_phylo_df(self, cfg):

        phylo_df = None

        p_score = np.load(self.phylo_path + "reduced_p_score.npy")
        feature_matrix = pd.read_pickle(self.feat_mat_whole + ".pkl")

        feature_matrix = self.interpret_helper_ob.update_feature_matrix(feature_matrix, p_score, mode="phylo")

        feature_matrix.to_pickle(self.feat_mat_phylo + ".pkl")

        logging.info("Pickle created for - chr : {}".format(str(self.chr_gc)))

        return feature_matrix


if __name__ == "__main__":
    setup_logging()
    dir = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/"
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/model"
    config_base = 'config.yaml'
    result_base = 'down_images'
    chr = 21
    mode = "lstm"

    logging.info("Get PE DFs")
    cfg = Interpretation.update_config(model_path, config_base, result_base)
    intp_ob = Interpretation(cfg, dir, chr, mode)

    # pe_df = intp_ob.get_pe_df(cfg)

    logging.info("Get GC DFs")
    gc_df, gc_content = intp_ob.get_gc_df(cfg)

    phylo_df = intp_ob.get_phylo_df(cfg)
