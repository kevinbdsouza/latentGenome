import logging
import numpy as np
import pandas as pd
from train_fns.test_gene import get_config
from keras.models import load_model
from downstream.avocado.run_avocado import AvocadoAnalysis

logger = logging.getLogger(__name__)


class AvoDownstreamHelper:
    def __init__(self, cfg):
        self.chr21_len = cfg.chr21_len
        self.cfg = cfg
        self.cfg_down = None
        self.columns = cfg.downstream_df_columns

    def create_mask(self, window_labels):
        ind_list = []
        label_ar = np.zeros(self.chr21_len)

        for i in range(window_labels.shape[0]):

            start = window_labels.loc[i, "start"]
            end = window_labels.loc[i, "end"]

            # print("gene : {} - start : {})".format(i, start))

            for j in range(end + 1 - start):
                ind_list.append(start - 1 + j)
                label_ar[start - 1 + j] = window_labels.loc[i, "target"]

        mask_vec = np.zeros(self.chr21_len, bool)
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
