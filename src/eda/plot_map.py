import logging
import numpy as np
from common.log import setup_logging
from train_fns.test_gene import get_config
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PlotMap:
    def __init__(self, cfg):
        self.chr21_len = cfg.chr21_len
        self.cfg = cfg
        self.path_h24 = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/all_ca_arc_sinh_h24/"
        self.path_h110 = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/all_ca_arc_sinh_h110/"

    def plot_all(self, path):
        rna_seq_dict, pe_dict, fire_dict, tad_dict = self.get_dict(path)

        self.plot_rna_seq(path, rna_seq_dict)
        self.plot_pe(path, pe_dict)
        self.plot_fire(path, fire_dict)
        self.plot_tad(path, tad_dict)

    def get_dict(self, path):
        rna_seq_dict = np.load(path + "map_dict_rnaseq.npy").item()
        pe_dict = np.load(path + "map_dict_pe.npy").item()
        fire_dict = np.load(path + "map_dict_fire.npy").item()
        tad_dict = np.load(path + "map_dict_tad.npy").item()

        return rna_seq_dict, pe_dict, fire_dict, tad_dict

    def get_lists(self, dict):
        key_list = []
        value_list = []
        for key, value in dict.items():
            key_list.append(key)
            value_list.append(value)

        return key_list, value_list

    def plot_rna_seq(self, path, rna_dict):
        key_list, value_list = self.get_lists(rna_dict)

        plt.figure()
        plt.scatter(key_list, value_list)
        plt.ylim(0, 1)
        plt.xticks(rotation=90, fontsize=6)
        plt.title('MAP for RNA-Seq prediction for different cell types')
        plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.savefig(path + 'rna_seq.png')

    def plot_pe(self, path, pe_dict):
        key_list, value_list = self.get_lists(pe_dict)

        plt.figure()
        plt.bar(range(len(key_list)), value_list, align='center')
        plt.xticks(range(len(key_list)), key_list)
        plt.title('MAP for PE interactions prediction for different cell types')
        plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.savefig(path + 'pe.png')

    def plot_fire(self, path, fire_dict):
        key_list, value_list = self.get_lists(fire_dict)

        plt.figure()
        plt.bar(range(len(key_list)), value_list, align='center')
        plt.xticks(range(len(key_list)), key_list)
        plt.title('MAP for FIRE prediction for different cell types')
        plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.savefig(path + 'fire.png')

    def plot_tad(self, path, tad_dict):
        key_list, value_list = self.get_lists(tad_dict)

        plt.figure()
        plt.bar(range(len(key_list)), value_list, align='center')
        plt.xticks(range(len(key_list)), key_list)
        plt.title('MAP for TAD prediction for different cell types')
        plt.xlabel('Cell Types')
        plt.ylabel('MAP')
        plt.savefig(path + 'tad.png')


if __name__ == "__main__":
    setup_logging()
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = "/home/kevindsouza/Documents/projects/latentGenome/results/04-03-2019_n/all_ca_arc_sinh_h24/model"

    cfg = get_config(model_path, config_base, result_base)
    plot_ob = PlotMap(cfg)

    plot_ob.plot_all(plot_ob.path_h24)
    plot_ob.plot_all(plot_ob.path_h110)

    print("done")