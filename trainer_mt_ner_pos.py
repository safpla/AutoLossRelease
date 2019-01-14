'''
The module for mulit-task training of machine translation, named entity
recognition and part of speech.
# __Author__ == 'Haowen Xu'
# __Date__ == '09-12-2018'
'''

import tensorflow as tf
import numpy as np
import logging
import os
import sys
import math
import socket
from time import gmtime, strftime
import time
from collections import deque

from models.s2s import rnn_attention_multi_heads
from models import controllers
from dataio.dataset import Dataset
import utils
from utils import config_util
from utils import data_utils
from utils.data_utils import prepare_batch
from utils.data_utils import prepare_train_batch
from utils import metrics
from utils import replaybuffer

root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()


class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, mode='TRAIN', exp_name=None):
        self.config = config

        hostname = socket.gethostname()
        hostname = '-'.join(hostname.split('.')[0:2])
        datetime = strftime('%m-%d-%H-%M', gmtime())
        if not exp_name:
            exp_name = '{}_{}'.format(hostname, datetime)
        logger.info('exp_name: {}'.format(exp_name))

        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto)
        self.model = rnn_attention_multi_heads.RNNAttentionMultiHeads(
            config,
            self.sess,
            mode,
            exp_name=exp_name,
            logger=logger
        )
        if config.controller_type == 'Fixed':
            self.controller = controllers.FixedController(config, self.sess)
        elif config.controller_type == 'MlpPPO':
            self.controller = controllers.MlpPPO(config, self.sess, exp_name+'controller')

        self.train_datasets = []
        self.valid_datasets = []
        self.test_datasets = []
        for task_num, task in enumerate(config.task_names):
            train_datasets = Dataset()
            valid_datasets = Dataset()
            test_datasets = Dataset()
            train_datasets.load_json(config.train_data_files[task_num])
            valid_datasets.load_json(config.valid_data_files[task_num])
            test_datasets.load_json(config.test_data_files[task_num])

            train_datasets.resize(int(train_datasets._num_examples / 4), shuffle=True)
            valid_datasets.resize(int(valid_datasets._num_examples / 4), shuffle=True)
            test_datasets.resize(int(test_datasets._num_examples / 4), shuffle=True)

            self.train_datasets.append(train_datasets)
            self.valid_datasets.append(valid_datasets)
            self.test_datasets.append(test_datasets)

    def train(self, load_model=None, save_model=False):
        config = self.config
        batch_size = config.batch_size
        start_time = time.time()
        model = self.model
        controller = self.controller
        model.initialize_weights()
        if load_model:
            model.load_model()

        history_train_loss = []
        history_train_acc = []
        history_valid_loss = []
        history_valid_acc = []
        best_acc = []
        history_len_task = config.history_len_task
        for i in range(len(config.task_names)):
            history_train_loss.append(deque(maxlen=history_len_task))
            history_train_acc.append(deque(maxlen=history_len_task))
            history_valid_loss.append(deque(maxlen=history_len_task))
            history_valid_acc.append(deque(maxlen=history_len_task))
            best_acc.append(0)

        no_impr_count = 0

        # Training loop
        logger.info('Start training...')
        for step in range(config.max_training_steps):
            task_to_train = controller.run_step(0, 0)[0]
            samples = self.train_datasets[task_to_train].next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)

            step_loss, step_acc = model.train(task_to_train,
                                              inputs, inputs_len,
                                              targets, targets_len,
                                              return_acc=True)
            history_train_loss[task_to_train].append(step_loss)
            history_train_acc[task_to_train].append(step_acc)

            if step % config.valid_frequency == 0:
                impr_flag = False
                for i in range(len(config.task_names)):
                    valid_loss, valid_acc = self.valid(i)
                    history_valid_loss[i].append(valid_loss)
                    history_valid_acc[i].append(valid_acc)
                    if best_acc[i] < valid_acc:
                        best_acc[i] = valid_acc
                        no_impr_count = 0
                        impr_flag = True
                if not impr_flag:
                    no_impr_count += 1

            if step % config.display_frequency == 0:
                self.display_training_process(history_train_loss,
                                              history_train_acc,
                                              history_valid_loss,
                                              history_valid_acc)

            if step % config.save_frequency == 0:
                logger.info('Saving the model...')
                model.save_model(step)

            if no_impr_count > config.endurance:
                break

    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        batch_size = config.batch_size
        controller = self.controller
        model = self.model
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBufferMeta = replaybuffer.ReplayBuffer(
            config.buffer_size, keys=keys)
        meta_training_history = deque(maxlen=10)
        epsilons = np.linspace(config.epsilon_start_meta,
                               config.epsilon_end_meta,
                               config.epsilon_decay_steps_meta)

        # ----Initialize controller.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)

        # ----Start meta loop.----
        for ep_meta in range(config.max_episodes_meta):
            logger.info('######## meta_episodes {} #########'.format(ep_meta))
            start_time = time.time()

            replayBufferMeta.clear()
            history_train_loss = []
            history_train_acc = []
            history_valid_loss = []
            history_valid_acc = []
            best_acc = []
            best_loss = []
            training_count = []
            history_len_task = config.history_len_task
            for i in range(len(config.task_names)):
                history_train_loss.append(deque(maxlen=history_len_task))
                history_train_acc.append(deque(maxlen=history_len_task))
                history_valid_loss.append(deque(maxlen=history_len_task))
                history_valid_acc.append(deque(maxlen=history_len_task))
                best_acc.append(0)
                best_loss.append(100)
                training_count.append(0)

            # ----Initialize task model.----
            model.initialize_weights()
            for i in range(len(config.task_names)):
                valid_loss, valid_acc = self.valid(i)
                history_valid_loss[i].append(valid_loss)
                history_valid_acc[i].append(valid_acc)
                history_train_loss[i].append(valid_loss)
                history_train_acc[i].append(valid_acc)

            update_count = 0
            no_impr_count = 0
            transitions = []
            epsilon = epsilons[min(ep_meta, config.epsilon_decay_steps_meta - 1)]
            meta_state = self.get_meta_state(history_train_loss,
                                             history_train_acc,
                                             history_valid_loss,
                                             history_valid_acc
                                            )
            for step in range(config.max_training_steps):
                meta_action = controller.run_step([meta_state], step, epsilon)[0]

                task_to_train = meta_action
                training_count[task_to_train] += 1
                #logger.info('task_to_train: {}'.format(task_to_train))
                samples = self.train_datasets[task_to_train].next_batch(batch_size)
                inputs = samples['input']
                targets = samples['target']
                inputs, inputs_len, targets, targets_len = prepare_train_batch(
                    inputs, targets, config.max_seq_length)

                step_loss, step_acc = model.train(task_to_train,
                                                  inputs, inputs_len,
                                                  targets, targets_len,
                                                  return_acc=True)
                history_train_loss[task_to_train].append(step_loss)
                history_train_acc[task_to_train].append(step_acc)

                meta_state_new = self.get_meta_state(history_train_loss,
                                                     history_train_acc,
                                                     history_valid_loss,
                                                     history_valid_acc
                                                     )
                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': None,
                              'next_state': meta_state_new,
                              'target_value': None}
                #print(transition)
                transitions.append(transition)

                if step > 0 and step % config.valid_frequency == 0:
                    impr_flag = False
                    for i in range(len(config.task_names)):
                        valid_loss, valid_acc = self.valid(i)
                        history_valid_loss[i].append(valid_loss)
                        history_valid_acc[i].append(valid_acc)
                        if best_loss[i] > valid_loss:
                            best_loss[i] = valid_loss
                            no_impr_count = 0
                            impr_flag = True
                    if not impr_flag:
                        no_impr_count += 1

                    # ----Online update of controller.----
                    update_count += 1
                    meta_reward = self.get_meta_reward_real_time(history_valid_loss, step)
                    for t in transitions:
                        t['reward'] = meta_reward
                        replayBufferMeta.add(t)

                    for _ in range(10):
                        lr = config.lr_meta
                        if replayBufferMeta.population == 0:
                            break
                        batch = replayBufferMeta.get_batch(min(config.batch_size_meta,
                                                               replayBufferMeta.population))
                        next_value = controller.get_value(batch['next_state'])
                        gamma = config.gamma_meta
                        batch['target_value'] = batch['reward'] + gamma * next_value
                        controller.update_actor(batch, lr)
                        controller.update_critic(batch, lr)

                    if update_count % config.sync_frequency_meta == 0:
                        controller.sync_net()
                        update_count = 0

                if step % config.display_frequency == 0:
                    logger.info('----Step: {:0>6d}----'.format(step))
                    self.display_training_process(history_train_loss,
                                                  history_train_acc,
                                                  history_valid_loss,
                                                  history_valid_acc)
                    logger.info('action distribution: {}'.format(
                        np.array(training_count) / sum(training_count)))
                    training_count = [0 for c in training_count]

                meta_state = meta_state_new

                #if self.check_terminate():
                #    break
                if no_impr_count > config.endurance:
                    logger.info(best_loss)
                    break
            if save_model:
                controller.save_model(ep_meta)
                model.save_model(0)
            #self.inference()

    def get_meta_state(self,
                       history_train_loss,
                       history_train_acc,
                       history_valid_loss,
                       history_valid_acc):
        state = []
        for i, task in enumerate(self.config.task_names):
            queue = history_train_loss[i]
            train_loss = np.mean(history_train_loss[i])
            valid_loss = history_valid_loss[i][-1]
            train_acc = np.mean(history_train_acc[i])
            valid_acc = history_valid_acc[i][-1]
            state.append((train_loss - 2) / 3)
            state.append((valid_loss - 2) / 3)
            state.append((train_loss - valid_loss))
            state.append(train_acc)
            state.append(valid_acc)
            state.append((train_acc - valid_acc) * 10)

        #for queue in history_train_loss:
        #    if len(queue) > 0:
        #        state.append(np.mean(queue))
        #    else:
        #        state.append(0)
        #for queue in history_train_acc:
        #    if len(queue) > 0:
        #        state.append(np.mean(queue))
        #    else:
        #        state.append(0)
        #for queue in history_valid_loss:
        #    if len(queue) > 0:
        #        state.append(queue[-1])
        #    else:
        #        state.append(0)
        #for queue in history_valid_acc:
        #    if len(queue) > 0:
        #        state.append(queue[-1])
        #    else:
        #        state.append(0)

        ## TODO: normalize
        #state[0] /= 10
        #state[2] /= 2
        #state[6] /= 10
        #state[8] /= 2
        return np.array(state)

    def get_meta_reward_real_time(self, historys, step):
        if len(historys[0]) < 2:
            return 0
        reward = 0
        reward += (historys[0][-2] - historys[0][-1]) * \
            (1 / max(historys[0][-1] - 4.6, 0.01))
        #print(reward)
        #print(historys[0][-1], historys[0][-2])
        #reward += (historys[1][-1] - historys[1][-2]) / 10
        #reward += (historys[2][-1] - historys[2][-2]) / 10
        return reward

    def valid(self, task_to_eval):
        config = self.config
        batch_size = config.batch_size
        valid_loss = 0.0
        valid_acc = 0.0
        valid_count = 0
        dataset = self.valid_datasets[task_to_eval]
        dataset.reset(shuffle=True)
        while dataset.epochs_completed < 1:
            samples = dataset.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)
            batch_loss, batch_acc = self.model.eval(task_to_eval,
                                                             inputs, inputs_len,
                                                             targets, targets_len,
                                                             return_acc=True)
            valid_loss += batch_loss * batch_size
            valid_acc += batch_acc * batch_size
            valid_count += batch_size

        valid_loss = valid_loss / valid_count
        valid_acc = valid_acc / valid_count
        return valid_loss, valid_acc

    def display_training_process(self,
                                 history_train_loss,
                                 history_train_acc,
                                 history_valid_loss,
                                 history_valid_acc):
        task_names = self.config.task_names
        for i, task in enumerate(task_names):
            train_loss = np.mean(history_train_loss[i])
            train_acc = np.mean(history_train_acc[i])
            valid_loss = history_valid_loss[i][-1]
            valid_acc = history_valid_acc[i][-1]
            logger.info('Task {:3}, loss: {:8.4f}|{:8.4f}, acc: {:8.4f}|{:8.4f}'.\
                        format(task, train_loss, valid_loss, train_acc, valid_acc))

    def inference(self, load_model=None):
        config = self.config
        batch_size = config.batch_size
        model = self.model
        if load_model:
            model.load_model(load_model)
        # Inference
        logger.info('Start inferencing...')
        dataset = self.valid_datasets[0]
        dataset.reset()
        correct_prediction = 0
        total_tokens = 0
        while dataset.epochs_completed < 1:
            samples = dataset.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len = prepare_batch(inputs)

            # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
            # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
            predicts = model.inference(encoder_inputs=inputs,
                                       encoder_inputs_length=inputs_len, task_num=0)
            targets = [[target] for target in targets]
            predicts = data_utils.unpaddle(predicts[:,:,0])
            result = metrics.compute_bleu(targets, predicts, 4)
            print(result)


if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info('Machine: {}'.format(socket.gethostname()))
    cfg_dir = os.path.join(root_path, 'config/mtNerPosRNNAttention/')
    config = config_util.load_config(cfg_dir)
    config.print_config(logger)

    # ----Instantiate a trainer object.----

    if argv[2] == 'train':
        # ----Training----
        logger.info('TRAIN')
        trainer = Trainer(config, exp_name=argv[1])
        load_model = '/media/haowen/AutoLossApps/saved_models/MlpPPO/'
        #trainer.train_meta(save_model=True, load_model=load_model)
        trainer.train_meta(save_model=True, load_model=False)

    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        trainer = Trainer(config, mode='DECODE', exp_name=argv[1])
        # TODO
    elif argv[2] == 'baseline_multi':
        # ----Baseline----
        logger.info('BASELINE')
    elif argv[2] == 'inference':
        # ----Generating----
        logger.info('INFERENCE')
        trainer = Trainer(config, mode='DECODE', exp_name=argv[1])
        load_model = '/media/haowen/AutoLossApps/saved_models/mt/'
        trainer.inference(load_model)
