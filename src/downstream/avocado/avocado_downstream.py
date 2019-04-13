import logging
from downstream.rna_seq import RnaSeq
from downstream.pe_interactions import PeInteractions
from train_fns.test_gene import get_config
from common.log import setup_logging
import numpy as np
import pandas as pd
from downstream.avocado.avo_down_helper import AvoDownstreamHelper
from downstream.downstream_helper import DownstreamHelper
from downstream.fires import Fires

gpu_id = 1
mode = "test"

logger = logging.getLogger(__name__)


class AvocadoDownstreamTasks:
    def __init__(self):
        self.rna_seq_path = "/opt/data/latent/data/downstream/RNA-seq"
        self.pe_int_path = "/opt/data/latent/data/downstream/PE-interactions"
        self.fire_path = "/opt/data/latent/data/downstream/FIREs"
        self.fire_cell_names = ['GM12878', 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']
        self.pe_cell_names = ['E123', 'E117', 'E116', 'E017']
        self.chr_list_rna = '21'
        self.chr_list_pe = 'chr21'
        self.chr_list_tad = 'chr21'
        self.chr_list_fire = 21
        self.saved_model_dir = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/avocado/model/"
        self.model_name = "avocado-chr21"
        self.Avo_downstream_helper_ob = AvoDownstreamHelper(cfg)
        self.downstream_helper_ob = DownstreamHelper(cfg)

    def run_rna_seq(self, cfg):

        rna_seq_ob = RnaSeq()
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_list_rna)
        rna_seq_chr['target'] = 0

        mean_map_dict = {}
        cls_mode = 'ind'
        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        for col in range(1, 58):
            rna_seq_chr.loc[rna_seq_chr.iloc[:, col] >= 0.5, 'target'] = 1
            rna_window_labels = rna_seq_chr.filter(['start', 'end', 'target'], axis=1)
            rna_window_labels = rna_window_labels.drop_duplicates(keep='first').reset_index(drop=True)
            rna_window_labels = rna_window_labels.drop([410, 598]).reset_index(drop=True)

            mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(rna_window_labels)

            avocado_features = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir, self.model_name,
                                                                                cfg)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(avocado_features, feature_matrix,
                                                                         mask_vector, label_ar)

            if feature_matrix["target"].value_counts()[0] > feature_matrix["target"].value_counts()[1]:
                bal_mode = "undersampling"
            else:
                bal_mode = "oversampling"

            feature_matrix = self.downstream_helper_ob.fix_class_imbalance(feature_matrix, mode=bal_mode)

            mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, cls_mode)

            mean_map_dict[rna_seq_chr.columns[col]] = mean_map

        np.save(self.saved_model_dir + 'map_dict_rnaseq.npy', mean_map_dict)

        return mean_map_dict

    def run_pe(self, cfg):

        pe_ob = PeInteractions()
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_list_pe)
        mean_map_dict = {}
        cls_mode = 'ind'
        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
            pe_window_labels = pe_data_chr_cell.filter(['window_start', 'window_end', 'label'], axis=1)
            pe_window_labels.rename(columns={'window_start': 'start', 'window_end': 'end', 'label': 'target'},
                                    inplace=True)
            pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(pe_window_labels)

            avocado_features = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir, self.model_name,
                                                                                cfg)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(avocado_features, feature_matrix,
                                                                         mask_vector, label_ar)

            if feature_matrix["target"].value_counts()[0] > feature_matrix["target"].value_counts()[1]:
                bal_mode = "undersampling"
            else:
                bal_mode = "oversampling"

            feature_matrix = self.downstream_helper_ob.fix_class_imbalance(feature_matrix, mode=bal_mode)

            mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, cls_mode)

            mean_map_dict[cell] = mean_map

        np.save(self.saved_model_dir + 'map_dict_pe.npy', mean_map_dict)

        return mean_map_dict

    def run_fires(self, cfg):

        fire_ob = Fires()
        fire_ob.get_fire_data(self.fire_path)
        fire_labeled = fire_ob.filter_fire_data(self.chr_list_fire)
        mean_map_dict = {}
        cls_mode = 'ind'
        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        for cell in self.fire_cell_names:
            fire_window_labels = fire_labeled.filter(['start', 'end', cell + '_l'], axis=1)
            fire_window_labels.rename(columns={cell + '_l': 'target'}, inplace=True)
            fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(fire_window_labels)

            avocado_features = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir, self.model_name,
                                                                                cfg)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(avocado_features, feature_matrix,
                                                                         mask_vector, label_ar)

            if feature_matrix["target"].value_counts()[0] > feature_matrix["target"].value_counts()[1]:
                bal_mode = "undersampling"
            else:
                bal_mode = "oversampling"

            feature_matrix = self.downstream_helper_ob.fix_class_imbalance(feature_matrix, mode=bal_mode)

            mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, cls_mode)

            mean_map_dict[cell] = mean_map

        np.save(self.saved_model_dir + 'map_dict_fire.npy', mean_map_dict)

        return mean_map_dict

    def run_tads(self, cfg):

        fire_ob = Fires()
        fire_ob.get_tad_data(self.fire_path, self.fire_cell_names)
        tad_filtered = fire_ob.filter_tad_data(self.chr_list_tad)
        mean_map_dict = {}
        cls_mode = 'ind'
        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        for col in range(7):
            tad_cell = tad_filtered[col]
            tad_cell['target'] = 1
            tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
            tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)
            tad_cell = fire_ob.augment_tad_negatives(cfg, tad_cell)

            mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(tad_cell)

            avocado_features = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir, self.model_name,
                                                                                cfg)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(avocado_features, feature_matrix,
                                                                         mask_vector, label_ar)

            if feature_matrix["target"].value_counts()[0] > feature_matrix["target"].value_counts()[1]:
                bal_mode = "undersampling"
            else:
                bal_mode = "oversampling"

            feature_matrix = self.downstream_helper_ob.fix_class_imbalance(feature_matrix, mode=bal_mode)

            mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, cls_mode)

            mean_map_dict[self.fire_cell_names[col]] = mean_map

        np.save(self.saved_model_dir + 'map_dict_tad.npy', mean_map_dict)

        return mean_map_dict


if __name__ == '__main__':
    setup_logging()
    config_base = 'avocado_config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/avocado/model"

    cfg = get_config(model_path, config_base, result_base)
    Av_downstream_ob = AvocadoDownstreamTasks()
    downstream_helper_ob = DownstreamHelper(cfg)

    mapdict_rna_seq = Av_downstream_ob.run_rna_seq(cfg)

    # mapdict_pe = downstream_ob.run_pe(cfg)

    # map_dict_fire = downstream_ob.run_fires(cfg)

    # map_dict_tad = downstream_ob.run_tads(cfg)

    print("done")
