""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
import math
from time import gmtime, strftime
import argparse

from models import controller_reinforce
import utils
from utils.analyse_utils import loss_analyzer_reg
from utils.analyse_utils import loss_analyzer_gan
import socket


root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description='run-time arguments', usage='')
    parser.add_argument('--task_name', required=True, type=str, help='task name')
    parser.add_argument('--task_mode', required=True, type=str, help='task mode')
    parser.add_argument('--exp_name', required=True, type=str, help='name of this experiment')
    parser.add_argument('--load_ctrl', type=str, help='load pretrained controller model')
    parser.add_argument('--save_ctrl', type=bool, default=True, help='whether to save the controller model')
    parser.add_argument('--test_trials', type=int, default=10, help='how many trials to run in test')
    parser.add_argument('--baseline_trials', type=int, default=10, help='how many trials to run in baseline')
    parser.add_argument('--lambda_task', type=float, help='coefficient for L1 regularization in regression and classification task')

    return parser.parse_args()

def arguments_checkout(config, args):
    # ----Task name checking.----
    if not args.task_name in ['reg', 'cls', 'gan', 'mnt', 'gan_cifar10']:
        logger.exception('Unexcepted task name')
        exit()

    # ----Mode checking.----
    if not args.task_mode in ['train', 'test', 'baseline', 'generate']:
        logger.exception('Unexcepted mode')
        exit()

    # ----Arguments in test mode.----
    if args.task_mode == 'test':
        if not args.load_ctrl:
            args.load_ctrl = os.path.join(config.model_dir, args.exp_name + '_ctrl')

def discount_rewards(transitions, final_reward):
    # TODO(haowen) Final reward + step reward
    for i in range(len(transitions)):
        transitions[i]['reward'] += final_reward


class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, args):
        self.config = config

        #hostname = config.hostname
        #hostname = '-'.join(hostname.split('.')[0:2])
        #datetime = strftime('%m-%d-%H-%M', gmtime())
        #exp_name = '{}_{}_{}'.format(hostname, datetime, args.exp_name)

        exp_name = args.exp_name
        logger.info('exp_name: {}'.format(exp_name))

        self.model_ctrl = controller_reinforce.Controller(config, exp_name+'_ctrl')
        if args.task_name == 'reg':
            from models import reg
            self.model_task = reg.Reg(config, exp_name+'_reg')
        elif args.task_name == 'cls':
            from models import cls
            self.model_task = cls.Cls(config, exp_name+'_cls')
        elif args.task_name == 'gan':
            from models import gan
            self.model_task = gan.Gan(config, exp_name+'_gan')
        elif args.task_name == 'gan_cifar10':
            from models import gan_cifar10
            self.model_task = gan_cifar10.Gan_cifar10(config,
                                                      exp_name+'_gan_cifar10')
        elif args.task_name == 'mnt':
            self.model_task = mnt.Mnt(config, exp_name+'_mnt')
        else:
            raise NotImplementedError

    def train(self, load_ctrl=None, save_ctrl=None):
        """ Iteratively training between controller and the task model """
        config = self.config
        lr_ctrl = config.lr_ctrl
        lr_task = config.lr_task
        model_ctrl = self.model_ctrl
        model_task = self.model_task
        # The bigger the performance is, the better.
        best_performance = -1e10
        # Record the number of latest episodes without a better result
        endurance = 0

        # ----Initialize controllor.----
        model_ctrl.initialize_weights()
        if load_ctrl:
            model_ctrl.load_model(load_ctrl)

        # TODO: package gradient related operations
        # ----Initialize gradient buffer.----
        gradBuffer = model_ctrl.get_weights()
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # ----Start episodes.----
        for ep in range(config.total_episodes):
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            # ----Initialize task model.----
            model_task.initialize_weights()
            model_task.reset()

            state = model_task.get_state()
            transitions = []

            step = -1
            # ----Running one episode.----
            logger.info('TRAINING TASK MODEL...')
            while True:
                step += 1
                action = model_ctrl.sample(state)
                state_new, reward, dead = model_task.response(action)
                extra_info = model_task.extra_info
                # ----Record training details.----
                transition = {'state': state,
                              'action': action,
                              'reward': reward,
                              'extra_info': extra_info}
                transitions.append(transition)

                state = state_new
                if dead:
                    break

            # Reinitialize the controller if the output of the controller
            # collapse to one specific action.
            if model_task.collapse:
                model_ctrl.initialize_weights()

            # ----Use the best model to get inception score on a
            #     larger number of samples to reduce the variance of reward----
            if args.task_name == 'gan' and model_task.checkpoint_dir:
                model_task.load_model(model_task.checkpoint_dir)
                # TODO use hyperparameter
                inps_test = model_task.get_inception_score(5000)
                logger.info('inps_test: {}'.format(inps_test))
                model_task.update_inception_score(inps_test[0])

            # ----Update the controller.----
            # We actually use the advantage as the final reward
            _, adv = model_task.get_final_reward()
            discount_rewards(transitions, adv)
            grads = model_ctrl.get_gradients(transitions)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if ep % config.update_frequency_ctrl == (config.update_frequency_ctrl - 1):
                logger.info('UPDATING CONTROLLOR...')
                model_ctrl.train_one_step(gradBuffer, lr_ctrl)

            # Printing training results and saving model_ctrl
            save_model_flag = False
            if model_task.best_performance > best_performance:
                best_performance = model_task.best_performance
                save_model_flag = True
                endurance = 0
            else:
                endurance += 1
            logger.info('----LOG FOR EPISODE {}----'.format(ep))
            logger.info('Performance in this episode: {}'.\
                        format(model_task.best_performance))
            logger.info('Best performance till now  : {}'.\
                        format(best_performance))

            if args.task_name == 'reg':
                loss_analyzer_reg(transitions)
            elif args.task_name == 'gan':
                loss_analyzer_gan(transitions)

            logger.info('--------------------------')

            if save_model_flag and save_ctrl:
                    model_ctrl.save_model(ep)
            if endurance > config.max_endurance_ctrl:
                break

    def test(self, load_ctrl, ckpt_num=None):
        config = self.config
        model_ctrl = self.model_ctrl
        model_task = self.model_task
        model_ctrl.initialize_weights()
        model_task.initialize_weights()
        model_ctrl.load_model(load_ctrl, ckpt_num=ckpt_num)
        model_task.reset()

        state = model_task.get_state()
        actions = np.array([0, 0])
        for i in range(config.max_training_step):
            action = model_ctrl.sample(state)
            actions += np.array(action)
            state_new, _, dead = model_task.response(action)
            state = state_new

        print(actions)

        if config.args.task_name == 'reg':
            model_task.load_model()
            loss, _, _ = model_task.valid(model_task.test_dataset)
            logger.info('test_loss: {}'.format(loss))
            return loss
        elif config.args.task_name == 'cls':
            raise NotImplementedError
        elif config.args.task_name == 'gan':
            raise NotImplementedError
        elif config.args.task_name == 'gan_grid':
            raise NotImplementedError
        elif config.args.task_name == 'gan_cifar10':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def baseline(self):
        config = self.config
        task_name = config.args.task_name
        if task_name == 'reg':
            model_ctrl = reg.controller_designed()
        model_task = self.model_task
        model_task.initialize_weights()
        model_task.reset()
        state = model_task.get_state()
        for i in range(config.max_training_step):
            action = model_ctrl.sample(state)
            state, _, dead = model_task.response(action)

        if config.args.task_name == 'reg':
            model_task.load_model()
            loss, _, _ = model_task.valid(model_task.test_dataset)
            logger.info('test_loss: {}, after {} steps of optimization'.\
                        format(loss, i))
            return loss

    def generate(self, load_stud):
        self.model_stud.initialize_weights()
        self.model_stud.load_model(load_stud)
        self.model_stud.generate_images(0)


if __name__ == '__main__':
    # ----Parsing arguments.----
    args = parse_args()

    # ----Loading config file.----
    logger.info('Machine name: {}'.format(socket.gethostname()))
    config_path = os.path.join(root_path, 'config/' + args.task_name)
    config = utils.load_config(config_path)

    arguments_checkout(config, args)
    utils.override_config(config, args)
    config.args = args

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, args)

    # ----Main entrance.----
    if args.task_mode == 'train':
        # ----Training----
        logger.info('TRAIN')
        trainer.train(load_ctrl=args.load_ctrl, save_ctrl=args.save_ctrl)
    elif args.task_mode == 'test':
        ## ----Testing----
        logger.info('TEST')
        test_metrics = []
        for i in range(args.test_trials):
            test_metrics.append(trainer.test(args.load_ctrl))
        logger.info(test_metrics)
    elif args.task_mode == 'baseline':
        # ----Baseline----
        logger.info('BASELINE')
        baseline_metrics = []
        for i in range(args.baseline_trials):
            baseline_metrics.append(trainer.baseline())
        logger.info(baseline_metrics)
    elif args.task_mode == 'generate':
        # ----Generate---- only in gan tasks
        logger.info('GENERATE')
        trainer.generate(args.load_stud)
