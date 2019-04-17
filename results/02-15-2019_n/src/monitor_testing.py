from __future__ import division
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MonitorTesting:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mse_iter = []
        self.mse_epgen = []

    def add_tf_summary(self, callback, loss_dict, seq_num):
        for name, value in loss_dict.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, seq_num)
            callback.writer.flush()

    def monitor_mse_iter(self, callback, mse, iter_num):
        self.mse_iter.append(mse)
        loss_dict_iter = {'mse_iter': mse}
        self.add_tf_summary(callback, loss_dict_iter, iter_num)

    def monitor_mse_epgen(self, callback, rec_loss, iter_num, epgen_num):
        self.monitor_mse_iter(callback, rec_loss, iter_num)

        epgen_mse = np.mean(self.mse_iter)
        self.mse_epgen.append(epgen_mse)
        loss_dict_epgen = {'mse_epgen': epgen_mse}
        self.add_tf_summary(callback, loss_dict_epgen, epgen_num)

        print('Mean Square Error for epgen is : {}'.format(epgen_mse))

        self.mse_iter = []
    