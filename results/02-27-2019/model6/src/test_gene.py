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
    hidden_states = np.zeros((2, cfg.hidden_size_encoder))
    # hidden_states = []
    encoder_init = True
    decoder_init = True

    try:
        for track_cut in data_gen_test:

            mse, hidden_states, encoder_init, predicted_cut, decoder_init = testing_loop(cfg, track_cut, model,
                                                                                         hidden_states, encoder_init,
                                                                                         decoder_init)

            iter_num += 1
            if iter_num % 500 == 0:
                logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                vizOb.plot_prediction(predicted_cut, track_cut, mse, iter_num)

            monitor.monitor_mse_iter(callback, np.sum(mse), iter_num)

    except Exception as e:
        logger.error(traceback.format_exc())

    logging.info('Testing complete')
    print('Mean MSE at end of testing: {}'.format(np.mean(monitor.mse_iter)))


def testing_loop(cfg, track_cut, model, hidden_states, encoder_init, decoder_init):
    encoder = model.encoder
    decoder = model.decoder

    if encoder_init:
        encoder_hidden, encoder_state = encoder.initHidden()
        encoder_init = False
    elif not encoder_init:
        encoder_hidden_init = hidden_states[0]
        # encoder_state_init = hidden_states[1]
        encoder_hidden = Variable(torch.from_numpy(encoder_hidden_init).float().unsqueeze(0).unsqueeze(0)).cuda()
        # encoder_state = Variable(torch.from_numpy(encoder_state_init).float().unsqueeze(0).unsqueeze(0)).cuda()
        _, encoder_state = encoder.initHidden()

    if decoder_init:
        decoder_hidden, _ = decoder.initHidden()
        decoder_init = False
    elif not decoder_init:
        decoder_hidden_init = hidden_states[1]
        # decoder_state_init = hidden_states[3]
        decoder_hidden = Variable(torch.from_numpy(decoder_hidden_init).float().unsqueeze(0).unsqueeze(0)).cuda()
        # decoder_state = Variable(torch.from_numpy(decoder_state_init).float().unsqueeze(0).unsqueeze(0)).cuda()

    input_variable = track_cut
    target_variable = track_cut
    nValues = target_variable.shape[1]

    encoder_outputs = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()
    encoder_hidden_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()
    encoder_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()

    mse = 0
    for ei in range(0, nValues):
        encoder_input = Variable(
            torch.from_numpy(np.array(input_variable[:, ei])).float().unsqueeze(0).unsqueeze(0)).cuda()

        encoder_output, encoder_hidden, encoder_state = encoder(encoder_input, encoder_hidden, encoder_state)
        encoder_outputs[ei] = encoder_output[0][0]
        encoder_hidden_states[ei] = encoder_hidden[0][0]
        encoder_states[ei] = encoder_state[0][0]

    hidden_states[0] = encoder_hidden.squeeze(0).cpu().data.numpy()
    # hidden_states[1] = encoder_state.squeeze(0).cpu().data.numpy()
    # decoder_hidden = encoder_hidden

    # With teacher forcing
    predicted_cut = np.zeros((cfg.input_size_encoder, cfg.cut_seq_len))

    for di in range(0, nValues):
        decoder_state = encoder_states[di].unsqueeze(0)
        decoder_output, decoder_hidden, _ = decoder(encoder_outputs[di].unsqueeze(0), decoder_hidden,
                                                    decoder_state)

        decoder_prediction = decoder_output.squeeze(0).cpu().data.numpy()
        predicted_cut[:, di] = decoder_prediction

        mse += np.power((decoder_prediction - target_variable[:, di]), 2)

    hidden_states[1] = decoder_hidden.squeeze(0).cpu().data.numpy()

    return mse / nValues, hidden_states, encoder_init, predicted_cut, decoder_init


if __name__ == '__main__':
    setup_logging()

    model_dir = '/home/kevindsouza/Documents/projects/latent/results/2019-02-27/model6/model'
    config_base = 'config.yaml'
    result_base = 'results'

    cfg = get_config(model_dir, config_base, result_base)

    test_gene(cfg)
