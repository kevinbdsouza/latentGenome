import logging
import numpy as np
import pandas as pd
from train_fns.test_gene import get_config
from keras.models import load_model
from downstream.avocado.run_avocado import AvocadoAnalysis
import yaml
import traceback

logger = logging.getLogger(__name__)


class AvoDownstreamHelper:
    def __init__(self, cfg):
        self.chr_len = cfg.chr_len
        self.cfg = cfg
        self.cfg_down = None
        self.columns = cfg.downstream_df_columns

    @staticmethod
    def save_config_as_yaml(path, cfg):
        """
            Save configuration
        """
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(cfg.__dict__, f, default_flow_style=False)
        except:
            logger.error(traceback.format_exc())

    def create_mask(self, window_labels):
        ind_list = []
        label_ar = np.zeros(self.chr_len)

        # w_labels = pd.read_pickle("/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-13/w_labels_dropped.pkl")

        for i in range(window_labels.shape[0]):

            start = window_labels.loc[i, "start"]
            end = window_labels.loc[i, "end"]

            # print("gene : {} - start : {})".format(i, start))

            if start > self.chr_len or end > self.chr_len:
                break

            for j in range(end + 1 - start):
                ind_list.append(start - 1 + j)
                label_ar[start - 1 + j] = window_labels.loc[i, "target"]

        mask_vec = np.zeros(self.chr_len, bool)
        ind_ar = np.array(ind_list)

        mask_vec[ind_ar] = True

        return mask_vec, label_ar

    def get_feature_matrix(self, model_path, model_name, cfg):

        Avocado_ob = AvocadoAnalysis()

        model = load_model("{}.h5".format(model_path + model_name))

        gen_factors = Avocado_ob.get_genomic_factors(model, cfg)

        return gen_factors

    def filter_states(self, avocado_features, feature_matrix, mask_vector, label_ar):

        # if True in mask_vector:
        #    print("here")

        enc = avocado_features[mask_vector,]
        lab = label_ar[mask_vector,]
        lab = lab.reshape((enc.shape[0], 1))

        feat_mat = np.append(enc, lab, axis=1)

        feature_matrix = feature_matrix.append(pd.DataFrame(feat_mat, columns=self.columns),
                                               ignore_index=True)

        feature_matrix.target = feature_matrix.target.astype(int)

        return feature_matrix


if __name__ == '__main__':
    file_name = "/home/kevindsouza/Documents/projects/latent/results/03-04-2019_n/downstream/feat_rna_h3_E004.pkl"
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latent/results/03-04-2019_n/downstream/h4/model"

    cfg = get_config(model_path, config_base, result_base)
    helper_ob = AvoDownstreamHelper(cfg)

    feature_matrix = pd.read_pickle(file_name)
    feature_matrix.target = feature_matrix.target.astype(int)
