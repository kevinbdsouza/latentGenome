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

gpu_id = 0
mode = "test"

logger = logging.getLogger(__name__)


class AvocadoDownstreamTasks:
    def __init__(self, model, chr, cfg, dir_name, mode):
        self.data_dir = "/data2/latent/data/"
        self.rna_seq_path = self.data_dir + "downstream/RNA-seq"
        self.pe_int_path = self.data_dir + "downstream/PE-interactions"
        self.fire_path = self.data_dir + "downstream/FIREs"
        self.fire_cell_names = ['GM12878', 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']
        self.pe_cell_names = ['E123', 'E117', 'E116', 'E017']
        self.feat_avo_rna = "feat_avo_chr_" + str(chr) + "_rna_"
        self.feat_avo_pe = "feat_avo_chr_" + str(chr) + "_pe_"
        self.feat_avo_fire = "feat_avo_chr_" + str(chr) + "_fire_"
        self.chr = chr
        self.chr_rna = str(chr)
        self.chr_pe = 'chr' + str(chr)
        self.chr_fire = chr
        self.saved_model_dir = dir_name
        self.model_name = model
        self.Avo_downstream_helper_ob = AvoDownstreamHelper(cfg)
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode=mode)

    def run_rna_seq(self, cfg):
        logging.info("Running RNA-Seq")

        rna_seq_ob = RnaSeq()
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_rna)
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

            gen_factors = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir,
                                                                           self.model_name,
                                                                           cfg, mask_vector)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(gen_factors, feature_matrix,
                                                                         mask_vector, label_ar)

            save_path = self.data_dir + 'avocado/chr' + str(self.chr) + "/" + self.feat_avo_rna + rna_seq_chr.columns[col]
            feature_matrix.to_pickle(save_path)
            logging.info("chr name : {} - cell name : {} - saved".format(str(self.chr), rna_seq_chr.columns[col]))

            # mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

            # mean_map_dict[rna_seq_chr.columns[col]] = mean_map

            # logging.info("cell name : {} - MAP : {}".format(rna_seq_chr.columns[col], mean_map))

        # np.save(self.saved_model_dir + 'map_dict_rnaseq.npy', mean_map_dict)

        return mean_map_dict

    def run_pe(self, cfg):
        logging.info("Running PE")

        pe_ob = PeInteractions()
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_pe)
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

            gen_factors = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir,
                                                                           self.model_name,
                                                                           cfg, mask_vector)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(gen_factors, feature_matrix,
                                                                         mask_vector, label_ar)

            save_path = self.data_dir + 'avocado/chr' + str(self.chr) + "/" + self.feat_avo_pe + cell
            feature_matrix.to_pickle(save_path)
            logging.info("chr name : {} - cell name : {} - saved".format(str(self.chr), cell))

            # mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

            # mean_map_dict[cell] = mean_map

            # logging.info("cell name : {} - MAP : {}".format(cell, mean_map))

        # np.save(self.saved_model_dir + 'map_dict_pe.npy', mean_map_dict)

        return mean_map_dict

    def run_fires(self, cfg):
        logging.info("Running FIREs")

        fire_ob = Fires()
        fire_ob.get_fire_data(self.fire_path)
        fire_labeled = fire_ob.filter_fire_data(self.chr_fire)
        mean_map_dict = {}
        cls_mode = 'ind'
        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        for cell in self.fire_cell_names:
            fire_window_labels = fire_labeled.filter(['start', 'end', cell + '_l'], axis=1)
            fire_window_labels.rename(columns={cell + '_l': 'target'}, inplace=True)
            fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar = self.Avo_downstream_helper_ob.create_mask(fire_window_labels)

            gen_factors = self.Avo_downstream_helper_ob.get_feature_matrix(self.saved_model_dir,
                                                                           self.model_name,
                                                                           cfg, mask_vector)

            feature_matrix = self.Avo_downstream_helper_ob.filter_states(gen_factors, feature_matrix,
                                                                         mask_vector, label_ar)

            save_path = self.data_dir + 'avocado/chr' + str(self.chr) + "/" + self.feat_avo_pe + cell
            feature_matrix.to_pickle(save_path)
            logging.info("chr name : {} - cell name : {} - saved".format(str(self.chr), cell))

            # mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

            # mean_map_dict[cell] = mean_map

            # logging.info("cell name : {} - MAP : {}".format(cell, mean_map))

        # np.save(self.saved_model_dir + 'map_dict_fire.npy', mean_map_dict)

        return mean_map_dict


if __name__ == '__main__':
    setup_logging()
    config_base = 'avocado_config.yaml'
    result_base = 'down_images'
    model_path = "/data2/latent/data/avocado"
    dir_name = "/data2/latent/data/avocado/"
    model = "avocado-chr21"
    chr = 21

    cfg = get_config(model_path, config_base, result_base)

    pd_col = list(np.arange(cfg.hidden_size_encoder))
    pd_col.append('target')
    cfg = cfg._replace(downstream_df_columns=pd_col)

    Av_downstream_ob = AvocadoDownstreamTasks(model, chr, cfg, dir_name, mode='avocado')

    # mapdict_rna_seq = Av_downstream_ob.run_rna_seq(cfg)

    mapdict_pe = Av_downstream_ob.run_pe(cfg)

    # map_dict_fire = Av_downstream_ob.run_fires(cfg)

    print("done")
