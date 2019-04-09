from __future__ import division
from train_fns import config
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_training import MonitorTraining
from train_fns.model import Model
import logging
from common.log import setup_logging
import traceback
import pandas as pd
import torch
from torch.autograd import Variable
from keras.callbacks import TensorBoard
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def train_iter_gene():
    cfg = config.Config()
    data_ob_gene = DataPrepGene(cfg, mode='train')

    data_ob_gene.prepare_id_dict()
    model = Model(cfg, data_ob_gene.vocab_size)
    if cfg.load_weights:
        model.load_weights()

    monitor = MonitorTraining(cfg, data_ob_gene.vocab_size)
    callback = TensorBoard(cfg.tensorboard_log_path)
    monitor.save_config_as_yaml(cfg.config_file, cfg)

    encoder_optimizer, decoder_optimizer, criterion = model.compile_optimizer()
    model.set_callback(callback)

    for epoch_num in range(cfg.num_epochs):

        logger.info('Epoch {}/{}'.format(epoch_num + 1, cfg.num_epochs))

        data_gen_train = data_ob_gene.get_data()

        iter_num = 0
        hidden_states = np.zeros((4, cfg.hidden_size_encoder))
        # hidden_states = []
        rec_loss = 0
        encoder_init = True
        decoder_init = True

        try:
            for track_cut in data_gen_train:

                rec_loss, hidden_states, encoder_init, decoder_init = training_loop(cfg, track_cut, model,
                                                                                    encoder_optimizer,
                                                                                    decoder_optimizer, criterion,
                                                                                    hidden_states, encoder_init,
                                                                                    decoder_init)

                iter_num += 1
                if iter_num % 500 == 0:
                    logger.info('Iter: {} - rec_loss: {}'.format(iter_num, np.mean(monitor.losses_iter)))
                    model.save_weights()

                monitor.monitor_loss_iter(callback, rec_loss, iter_num)

            save_flag = monitor.monitor_loss_epoch(callback, rec_loss, iter_num, epoch_num)
            if save_flag == 'True':
                model.save_weights()

        except Exception as e:
            logger.error(traceback.format_exc())
            model.save_weights()
            continue

    model.save_weights()
    logging.info('Training complete, exiting.')


def training_loop(cfg, track_cut, model, encoder_optimizer,
                  decoder_optimizer, criterion, hidden_states, encoder_init, decoder_init):
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

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = track_cut
    target_variable = track_cut
    nValues = target_variable.shape[1]

    encoder_outputs = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()
    encoder_hidden_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()
    encoder_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda()

    rec_loss = 0
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
    for di in range(0, nValues):
        decoder_state = encoder_states[di].unsqueeze(0)
        decoder_output, decoder_hidden, _ = decoder(encoder_outputs[di].unsqueeze(0), decoder_hidden,
                                                    decoder_state)

        decoder_target = Variable(
            torch.from_numpy(np.array(target_variable[:, di])).float().unsqueeze(0)).cuda()

        # decoder_input = Variable(next_input.type(torch.FloatTensor).unsqueeze(0)).cuda()

        rec_loss += criterion(decoder_output.squeeze(0), decoder_target)

    hidden_states[1] = decoder_hidden.squeeze(0).cpu().data.numpy()
    # hidden_states[3] = decoder_state.squeeze(0).cpu().data.numpy()

    rec_loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return rec_loss.item() / nValues, hidden_states, encoder_init, decoder_init


if __name__ == '__main__':
    setup_logging()
    train_iter_gene()
