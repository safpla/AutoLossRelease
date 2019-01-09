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
from models import reg
#from models import cls
#from models import gan
#from models import gan_grid
#from models import gan_cifar10
import utils
from utils.analyse_utils import loss_analyzer_toy
#from utils.analyse_utils import loss_analyzer_gan
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
            self.model_task = reg.Reg(config, exp_name+'_reg')
        elif args.task_name == 'cls':
            self.model_task = cls.Cls(config, exp_name+'_cls')
        elif args.task_name == 'gan':
            self.model_task = gan.Gan(config, exp_name+'_gan', arch=arch)
        elif args.task_name == 'gan_cifar10':
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
        best_reward = -1e5 # A big negative initial value
        best_acc = 0
        best_loss = 0
        best_inps = 0
        best_best_inps = 0
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

        # ----The epsilon decay schedule.----
        epsilons = np.linspace(config.epsilon_start_ctrl,
                               config.epsilon_end_ctrl,
                               config.epsilon_decay_steps_ctrl)

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
            epsilon = epsilons[min(ep, config.epsilon_decay_steps_ctrl-1)]
            logger.info('epsilon={}'.format(epsilon))
            # ----Running one episode.----
            while True:
                step += 1
                action = model_ctrl.sample(state, explore_rate=epsilon)
                state_new, reward, dead = model_task.response(action)
                # ----Record training details.----
                transition = {'state': state,
                              'action': action,
                              'reward': reward}
                transitions.append(transition)

                # TODO delete
                if 'gan' in args.task_name:
                    transition['gen_cost'] = model_task.ema_gen_cost
                    transition['disc_cost_real'] = model_task.ema_disc_cost_real
                    transition['disc_cost_fake'] = model_task.ema_disc_cost_fake
                else:
                    transition['valid_loss'] = model_task.previous_valid_loss[-1]
                    transition['train_loss'] = model_task.previous_train_loss[-1]

                state = state_new
                if dead:
                    break

            # ----Use the best model to get inception score on a
            #     larger number of samples to reduce the variance of reward----
            if args.task_name == 'gan' and model_task.checkpoint_dir:
                model_task.load_model(model_task.checkpoint_dir)
                # TODO use hyperparameter
                inps_test = model_task.get_inception_score(5000)
                logger.info('inps_test: {}'.format(inps_test))
                model_task.update_inception_score(inps_test[0])
            else:
                logger.info('task model hasn\'t been saved before')

            # ----Update the controller.----
            final_reward, adv = model_task.get_final_reward()
            discount_rewards(transitions, adv)
            grads = model_ctrl.get_gradients(transitions)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if ep % config.update_frequency_ctrl == (config.update_frequency_ctrl - 1):
                logger.info('UPDATE CONTROLLOR')
                logger.info('lr_ctrl: {}'.format(lr_ctrl))
                model_ctrl.train_one_step(gradBuffer, lr_ctrl)

            save_model_flag = False
            if args.task_name == 'reg':
                loss_analyzer_toy(transitions)
                loss = model_task.best_loss
                # compare results after enough exploration.
                if ep > config.epsilon_decay_steps_ctrl:
                    if final_reward > best_reward:
                        best_reward = final_reward
                        best_loss = loss
                        save_model_flag = True
                        endurance = 0
                    else:
                        endurance += 1
                logger.info('best_loss: {}'.format(loss))
                logger.info('lambda: {}'.format(config.lambda_task))
            elif args.task_name == 'cls':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                acc = model_task.best_acc
                loss = model_task.best_loss
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_acc = acc
                    best_loss = loss
                    save_model_flag = True
                    enduranc = 0
                else:
                    endurance += 1
                logger.info('acc: {}'.format(acc))
                logger.info('best_acc: {}'.format(best_acc))
                logger.info('best_loss: {}'.format(loss))
            elif args.task_name == 'gan' or\
                args.task_name == 'gan_cifar10':
                loss_analyzer_gan(action_hist, reward_hist)
                best_inps = model_task.best_inception_score
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_best_inps = best_inps
                    save_model_flag = True
                logger.info('best_inps: {}'.format(best_inps))
                logger.info('best_best_inps: {}'.format(best_best_inps))
                logger.info('final_inps_baseline: {}'.\
                            format(model_task.final_inps_baseline))
            elif args.task_name == 'gan_grid':
                loss_analyzer_gan(action_hist, reward_hist)
                hq = model_task.best_hq
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_hq = hq
                    save_model_flag = True
                logger.info('hq: {}'.format(hq))
                logger.info('best_hq: {}'.format(best_hq))

            logger.info('adv: {}'.format(adv))

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
            action = model_ctrl.sample(state, 0.1)
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
