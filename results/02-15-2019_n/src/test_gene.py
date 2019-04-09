from __future__ import division
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_testing import MonitorTesting
from eda.viz import Viz
from train_fns.model import Model
import logging
from common.log import setup_logging
from common import utils
import traceback
import torch
from torch.autograd import Variable
from keras.callbacks import TensorBoard
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def get_config(model_dir, config_base, result_base):
    encoder_path = os.path.join(model_dir, '/encoder.pth')
    decoder_path = os.path.join(model_dir, '/decoder.pth')
    config_path = os.path.join(model_dir, config_base)
    res_path = os.path.join(model_dir, result_base)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    cfg = utils.load_config_as_class(model_dir, config_path, encoder_path, decoder_path, res_path)
    return cfg


def test_gene(cfg):
    data_ob_gene = DataPrepGene(cfg, mode='test')
    monitor = MonitorTesting(cfg)
    callback = TensorBoard(cfg.tensorboard_log_path)
    vizOb = Viz(cfg)

    data_ob_gene.prepare_id_dict()
    data_gen_test = data_ob_gene.get_data()
    model = Model(cfg, data_ob_gene.vocab_size)
    model.load_weights()
    model.set_callback(callback)

    logger.info('Testing Start')

    iter_num = 0
    prev_epgen_id = None
    hidden_states = []
    epgen_num = 0

    try:
        for epgen_assay_id, track_cut in data_gen_test:

            log_mode = 'iter'
            if prev_epgen_id != epgen_assay_id:
                print('New epgen-assay:{} with ID:{}'.format(data_ob_gene.inv_epgen_dict[epgen_assay_id],
                                                             epgen_assay_id))
                log_mode = 'epgen'

            mse, hidden_states, prev_epgen_id, predicted_cut = testing_loop(cfg, track_cut, model,
                                                                            hidden_states,
                                                                            epgen_assay_id, prev_epgen_id)

            iter_num += 1
            if iter_num % 500 == 0:
                logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                vizOb.plot_prediction(predicted_cut, track_cut, mse, iter_num, epgen_num)

            if epgen_assay_id != 0 and log_mode == 'epgen':
                monitor.monitor_mse_epgen(callback, mse, iter_num, epgen_num)
                epgen_num += 1
            elif log_mode == 'iter':
                monitor.monitor_mse_iter(callback, mse, iter_num)

    except Exception as e:
        logger.error(traceback.format_exc())

    logging.info('Testing complete')
    print('Mean MSE at end of testing: {}'.format(np.mean(monitor.mse_epgen)))


def testing_loop(cfg, track_cut, model, hidden_states, epgen_assay_id, prev_epgen_id):
    encoder = model.encoder
    decoder = model.decoder
    ca_embedder = model.ca_embedder

    # create ca embedding
    ca_embedding = ca_embedder.celltype_assay_embedding(torch.tensor(epgen_assay_id).cuda())
    ca_embedding = ca_embedding.unsqueeze(0).unsqueeze(0)

    if prev_epgen_id != epgen_assay_id:
        encoder_hidden, encoder_state = encoder.initHidden()
        prev_epgen_id = epgen_assay_id
    else:
        encoder_init = hidden_states[0, :50]
        encoder_hidden = Variable(torch.from_numpy(encoder_init).float().unsqueeze(0).unsqueeze(0)).cuda()
        _, encoder_state = encoder.initHidden()

    input_variable = track_cut
    target_variable = track_cut
    nValues = target_variable.shape[0]

    encoder_outputs = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()
    encoder_hidden_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()

    mse = 0
    for ei in range(0, nValues):
        encoder_input = Variable(
            torch.from_numpy(np.array(input_variable[ei])).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)).cuda()

        encoder_output, encoder_hidden, encoder_state = encoder(encoder_input, encoder_hidden, encoder_state)
        encoder_outputs[ei] = encoder_output[0][0]
        encoder_hidden_states[ei] = encoder_hidden[0][0]

    encoder_store = encoder_hidden.squeeze(0).cpu().data.numpy()
    ca_store = ca_embedding.cpu().squeeze(0).data.numpy()
    hidden_states = np.concatenate((encoder_store, ca_store), axis=1)

    decoder_hidden = torch.cat((encoder_hidden, ca_embedding), 2)

    # With teacher forcing
    predicted_cut = []
    for di in range(0, nValues):
        decoder_state = torch.cat((encoder_hidden_states[di].unsqueeze(0), ca_embedding), 2)
        decoder_output, decoder_hidden, _ = decoder(encoder_outputs[di].unsqueeze(0), decoder_hidden,
                                                    decoder_state)

        decoder_prediction = decoder_output.squeeze(0).cpu().data.numpy()
        predicted_cut.append(decoder_prediction[0][0])

        mse += np.power((decoder_prediction - target_variable[di]), 2)

    return mse / nValues, hidden_states, prev_epgen_id, np.array(predicted_cut)


if __name__ == '__main__':
    setup_logging()

    model_dir = '/home/kevindsouza/Documents/projects/latent/src/saved_model'
    config_base = 'config.yaml'
    result_base = 'results'

    cfg = get_config(model_dir, config_base, result_base)

    test_gene(cfg)
