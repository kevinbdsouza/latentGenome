from downstream.avocado.avocado_downstream import AvocadoDownstreamTasks
from train_fns.test_gene import get_config
import numpy as np
import logging
from common.log import setup_logging
import json

logger = logging.getLogger(__name__)


def get_avg_map(mapdict_rna_seq):
    map_vec = []

    for k, v in mapdict_rna_seq.items():
        map_vec.append(v)

    mean_map = np.array(map_vec).mean()

    return mean_map


def run_all(model, down_dir, chr_list):
    config_base = 'avocado_config.yaml'
    result_base = 'down_images'

    cfg = get_config(down_dir, config_base, result_base)
    pd_col = list(np.arange(cfg.hidden_size_encoder))
    pd_col.append('target')
    pd_col.append('gene_id')
    cfg = cfg._replace(downstream_df_columns=pd_col)

    map_rna = []
    map_pe = []
    map_fire = []
    for chr in chr_list:
        dir_name = down_dir + "/" + "chr" + str(chr) + "/"
        file_name = down_dir + "/" + "chr" + str(chr) + "/" + model + str(chr)
        model_name = model + str(chr)

        with open(file_name + '.json') as json_file:
            data = json.load(json_file)
            chr_len = data["n_genomic_positions"]

        cfg = cfg._replace(chr_len=chr_len)

        Av_downstream_ob = AvocadoDownstreamTasks(model_name, chr, cfg, dir_name, mode='avocado')

        mapdict_rna_seq = {}
        # mapdict_rna_seq = Av_downstream_ob.run_rna_seq(cfg)
        # mean_rna_map = get_avg_map(mapdict_rna_seq)
        # map_rna.append(mean_rna_map)

        # mapdict_pe = {}
        mapdict_pe = Av_downstream_ob.run_pe(cfg)
        # mean_pe_map = get_avg_map(mapdict_pe)
        # map_pe.append(mean_pe_map)

        mapdict_fire = {}
        # mapdict_fire = Av_downstream_ob.run_fires(cfg)
        # mean_fire_map = get_avg_map(mapdict_fire)
        # map_fire.append(mean_fire_map)

        # logging.info("chr: {} - map_rna:{}".format(chr, mean_rna_map))
        # logging.info("chr: {} - map_pe: {}".format(chr, mean_pe_map))
        # logging.info("chr: {} - map_fire: {}".format(chr, mean_fire_map))

    return mapdict_rna_seq, mapdict_pe, mapdict_fire


if __name__ == "__main__":
    # setup_logging()
    logging.basicConfig(
        filename="/home/kevindsouza/Documents/projects/latentGenome/src/saved_model/avocado/avocado_log.txt",
        level=logging.DEBUG)

    model = "avocado-chr"
    down_dir = "/data2/latent/data/avocado"
    chr_list = np.arange(20, 21)

    map_rna, map_pe, map_fire = run_all(model, down_dir, chr_list)

    np.save(down_dir + "/" + "map_rna.npy", map_rna)
    np.save(down_dir + "/" + "map_pe.npy", map_pe)
    np.save(down_dir + "/" + "map_fire.npy", map_fire)

    print("done")
