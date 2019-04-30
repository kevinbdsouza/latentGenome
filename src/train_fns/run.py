from train_fns import train_gene
from downstream.run_downstream import DownstreamTasks
from train_fns import config
import os
from train_fns.test_gene import get_config
import numpy as np


def get_avg_map(mapdict_rna_seq):
    map_vec = []

    for k, v in mapdict_rna_seq.iteritems():
        map_vec.append(v)

    mean_map = np.array(map_vec).mean()

    return mean_map


def run_all(hidden_nodes, cfg, down_dir):
    config_base = 'config.yaml'
    result_base = 'down_images'
    map_list_hidden = []

    for h in hidden_nodes:
        dir_name = down_dir + "/" + str(h) + "/"
        model_dir_name = dir_name + "model"

        cfg = cfg._replace(hidden_size_encoder=h)

        train_gene.train_iter_gene(cfg)

        os.system("cd {}".format(down_dir))
        os.system("mkdir {}".format(h))
        os.system("cd {}".format(h))
        os.system("mkdir model")
        os.system("mv - v {}/* {}/".format(cfg.model_dir, model_dir_name))

        cfg = get_config(model_dir_name, config_base, result_base)
        pd_col = list(np.arange(cfg.hidden_size_encoder))
        pd_col.append('target')
        pd_col.append('gene_id')
        cfg = cfg._replace(downstream_df_columns=pd_col)

        downstream_ob = DownstreamTasks(dir_name)

        mapdict_rna_seq = downstream_ob.run_rna_seq(cfg)

        map_list_hidden.append(get_avg_map(mapdict_rna_seq))

    return map_list_hidden


if __name__ == "__main__":
    cfg = config.Config()
    hidden_nodes = [6, 12, 24, 48, 96, 110]
    down_dir = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/hidden"

    map_list_hidden = run_all(hidden_nodes, cfg, down_dir)

    np.save(down_dir + "/" + "map_hidden.npy", map_list_hidden)

    print("done")
