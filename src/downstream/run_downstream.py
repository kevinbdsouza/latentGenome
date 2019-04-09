import logging
from downstream.rna_seq import RnaSeq
from downstream.pe_interactions import PeInteractions
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_testing import MonitorTesting
from train_fns.train_gene import unroll_loop
from train_fns.test_gene import get_config
from common.log import setup_logging
from keras.callbacks import TensorBoard
from eda.viz import Viz
from train_fns.model import Model
import traceback
import numpy as np
import pandas as pd
from downstream.downstream_helper import DownstreamHelper
from downstream.downstream_lstm import DownstreamLSTM
from downstream.fires import Fires

gpu_id = 1
mode = "test"

logger = logging.getLogger(__name__)


class DownstreamTasks:
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
        self.saved_model_dir = "/home/kevindsouza/Documents/projects/latent/results/04-03-2019_n/all_ca_arc_sinh/"
        self.feat_mat_rna = self.saved_model_dir + "feat_rna_h24_E003.pkl"
        self.feat_mat_pe = self.saved_model_dir + "feat_pe_h24_E117.pkl"
        self.feat_mat_fire = self.saved_model_dir + "feat_fire_h24_H1.pkl"
        self.feat_mat_tad = self.saved_model_dir + "feat_tad_h24_GM.pkl"
        self.new_features = self.saved_model_dir + "new_feat.npy"
        self.run_features_rna = False
        self.run_features_pe = True
        self.run_features_fire = False
        self.run_features_tad = True
        self.concat_lstm = False
        self.run_concat_feat = False
        self.downstream_helper_ob = DownstreamHelper(cfg)
        self.down_lstm_ob = DownstreamLSTM()

    def downstream_main(self, cfg, mask_vector, label_ar, gene_ar):
        data_ob_gene = DataPrepGene(cfg, mode='test')
        monitor = MonitorTesting(cfg)
        callback = TensorBoard(cfg.tensorboard_log_path)

        data_ob_gene.prepare_id_dict()
        data_gen_test = data_ob_gene.get_data()
        model = Model(cfg, data_ob_gene.vocab_size, gpu_id)
        model.load_weights()
        model.set_callback(callback)

        logger.info('Downstream Start')

        iter_num = 0
        hidden_states = np.zeros((2, cfg.hidden_size_encoder))

        encoder_init = True
        decoder_init = True
        encoder_optimizer, decoder_optimizer, criterion = None, None, None

        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        try:
            for track_cut in data_gen_test:

                mse, hidden_states, encoder_init, decoder_init, predicted_cut, encoder_hidden_states_np = unroll_loop(
                    cfg, track_cut, model,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                    hidden_states, encoder_init,
                    decoder_init, mode)

                mask_vector_cut = mask_vector[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]
                label_ar_cut = label_ar[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]
                gene_ar_cut = gene_ar[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]

                feature_matrix = self.downstream_helper_ob.filter_states(encoder_hidden_states_np, feature_matrix,
                                                                         mask_vector_cut,
                                                                         label_ar_cut, gene_ar_cut)

                iter_num += 1

                if iter_num % 500 == 0:
                    logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                    # vizOb.plot_prediction(predicted_cut, track_cut, mse, iter_num)

                monitor.monitor_mse_iter(callback, np.sum(mse), iter_num)

        except Exception as e:
            logger.error(traceback.format_exc())

        print('Mean MSE at end of Downstream Run: {}'.format(np.mean(monitor.mse_iter)))

        return feature_matrix

    def run_rna_seq(self, cfg):

        rna_seq_ob = RnaSeq()
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_list_rna)

        rna_seq_chr['target'] = 0
        mean_map_dict = {}

        for col in range(1, 58):
            rna_seq_chr.loc[rna_seq_chr.iloc[:, col] >= 0.5, 'target'] = 1
            rna_window_labels = rna_seq_chr.filter(['start', 'end', 'target'], axis=1)
            rna_window_labels = rna_window_labels.drop_duplicates(keep='first').reset_index(drop=True)
            rna_window_labels = rna_window_labels.drop([410, 598]).reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(rna_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_rna,
                                                                          self.feat_mat_rna, self.downstream_main)

            if self.concat_lstm:

                if self.run_concat_feat:
                    new_feature_mat = self.downstream_helper_ob.concat_gene_features(feature_matrix)
                    new_feature_mat.to_pickle(self.new_features)
                else:
                    new_feature_mat = np.load(self.new_features)

                target = np.array(rna_window_labels[:]["target"])
                feature_matrix, cfg_down = self.down_lstm_ob.get_features(new_feature_mat, target)
                self.downstream_helper_ob.cfg.down = cfg_down
                cls_mode = 'concat'
            else:
                feature_matrix = feature_matrix.loc[:, feature_matrix.columns != 'gene_id']
                cls_mode = 'ind'

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

        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
            pe_window_labels = pe_data_chr_cell.filter(['window_start', 'window_end', 'label'], axis=1)
            pe_window_labels.rename(columns={'window_start': 'start', 'window_end': 'end', 'label': 'target'},
                                    inplace=True)
            pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(pe_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_pe,
                                                                          self.feat_mat_pe, self.downstream_main)

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

        for cell in self.fire_cell_names:
            fire_window_labels = fire_labeled.filter(['start', 'end', cell + '_l'], axis=1)
            fire_window_labels.rename(columns={cell + '_l': 'target'}, inplace=True)
            fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(fire_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_fire,
                                                                          self.feat_mat_fire, self.downstream_main)

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

        for col in range(7):
            tad_cell = tad_filtered[col]
            tad_cell['target'] = 1
            tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
            tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)
            tad_cell = fire_ob.augment_tad_negatives(cfg, tad_cell)
            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(tad_cell)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_tad,
                                                                          self.feat_mat_tad, self.downstream_main)

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
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latent/results/04-03-2019_n/all_ca_arc_sinh/model"

    cfg = get_config(model_path, config_base, result_base)
    downstream_ob = DownstreamTasks()
    downstream_helper_ob = DownstreamHelper(cfg)

    mapdict_rna_seq = downstream_ob.run_rna_seq(cfg)

    mapdict_pe = downstream_ob.run_pe(cfg)

    map_dict_fire = downstream_ob.run_fires(cfg)

    map_dict_tad = downstream_ob.run_tads(cfg)

    print("done")
