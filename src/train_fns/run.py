from train_fns import train_gene
from downstream.run_downstream import DownstreamTasks
from train_fns import config
import os
from train_fns.test_gene import get_config
import numpy as np
import logging
from common.log import setup_logging

logger = logging.getLogger(__name__)


def get_avg_map(mapdict_rna_seq):
    map_vec = []

    for k, v in mapdict_rna_seq.items():
        map_vec.append(v)

    mean_map = np.array(map_vec).mean()

    return mean_map


def run_all(hidden_nodes, down_dir):
    config_base = 'config.yaml'
    result_base = 'down_images'
    map_list_hidden = []

    for h in hidden_nodes:
        cfg = config.Config()
        dir_name = down_dir + "/" + str(5e-10) + "/"
        model_dir_name = dir_name + "model"

        cfg.hidden_size_encoder = h
        cfg.input_size_decoder = h
        cfg.hidden_size_decoder = h

        train_gene.train_iter_gene(cfg)

        os.system("mkdir {}".format(dir_name))
        os.system("mkdir {}".format(model_dir_name))
        os.system("mv -v {}/* {}/".format(cfg.model_dir, model_dir_name))

        cfg = get_config(model_dir_name, config_base, result_base)
        pd_col = list(np.arange(cfg.hidden_size_encoder))
        pd_col.append('target')
        pd_col.append('gene_id')
        cfg = cfg._replace(downstream_df_columns=pd_col)

        downstream_ob = DownstreamTasks(cfg, dir_name)

        mapdict_rna_seq, feature_matrix = downstream_ob.run_rna_seq(cfg)

        logging.info("hidden nodes: {}".format(h))
        logging.info("feature_matrix: {}".format(feature_matrix))
        logging.info("mapdict_rna_seq: {}".format(mapdict_rna_seq))

        map_list_hidden.append(get_avg_map(mapdict_rna_seq))

    return map_list_hidden


if __name__ == "__main__":
    # setup_logging()
    logging.basicConfig(filename="run_log.txt",
                        level=logging.DEBUG)

    hidden_nodes = [110]
    # hidden_nodes = [6, 12, 24, 36, 48, 60, 96, 110]
    down_dir = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110"

    map_list_hidden = run_all(hidden_nodes, down_dir)

    np.save(down_dir + "/" + "map_hidden.npy", map_list_hidden)

    print("done")
