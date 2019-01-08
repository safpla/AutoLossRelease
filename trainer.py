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

#from models import controller
#from models import reg
#from models import cls
#from models import gan
#from models import gan_grid
#from models import gan_cifar10
import utils
#from utils.analyse_utils import loss_analyzer_toy
#from utils.analyse_utils import loss_analyzer_gan
#import socket


root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description='run-time arguments', usage='')
    parser.add_argument('--task_name', required=True, type=str, help='task name')
    parser.add_argument('--task_mode', required=True, type=str, help='task mode')
    parser.add_argument('--exp_name', required=True, type=str, help='name of this experiment')
    parser.add_argument('--load_ctrl', type=str, help='load pretrained controller model')
    parser.add_argument('--save_ctrl', type=bool, default=False, help='whether to save the controller model')
    parser.add_argument('--test_trials', type=int, default=10, help='how many trials to run in test')
    parser.add_argument('--baseline_trials', type=int, default=10, help='how many trials to run in baseline')

    return parser.parse_args()

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

        self.model_ctrl = controller.Controller(config, exp_name+'_ctrl')
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
        lr = config.lr_ctrl
        model_ctrl = self.model_ctrl
        model_task = self.model_task_
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
            while True:
                step += 1
                action = model_ctrl.sample(state)
                state_new, reward, dead = model_task.response(action)
                # ----Record training details.----
                transition = {'state': state,
                              'action': action,
                              'reward': reward}
                transitions.append(transition)

                # TODO delete
                #if 'gan' in args.task_name:
                #    gen_cost_hist.append(model_task.ema_gen_cost)
                #    disc_cost_real_hist.append(model_task.ema_disc_cost_real)
                #    disc_cost_fake_hist.append(model_task.ema_disc_cost_fake)
                #else:
                #    valid_loss_hist.append(model_task.previous_valid_loss[-1])
                #    train_loss_hist.append(model_task.previous_train_loss[-1])

                state = state_new
                if dead:
                    break

            # ----For Gan, use the best model to get inception score on a
            #     larger number of samples to reduce the variance of reward----
            if args.task_name == 'gan' and model_task.task_dir:
                model_task.load_model(model_task.task_dir)
                # TODO use hyperparameter
                inps_test = model_task.get_inception_score(5000)
                logger.info('inps_test: {}'.format(inps_test))
                model_task.update_inception_score(inps_test[0])
            else:
                logger.info('task model hasn\'t been saved before')

            # ----Update the controller.----
            final_reward, adv = model_task.get_final_reward()
            print(transitions)
            reward_hist = discount_rewards(transitions, adv)
            print(transitions)
            grads = model_ctrl.get_gradients(transitions)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if lr > config.lr_rl * 0.1:
                lr = lr * config.lr_decay_rl
            if ep % config.update_frequency == (config.update_frequency - 1):
                logger.info('UPDATE CONTROLLOR')
                logger.info('lr_ctrl: {}'.format(lr))
                model_ctrl.train_one_step(gradBuffer, lr)
                #logger.info('grad')
                #for ix, grad in enumerate(gradBuffer):
                #    logger.info(grad)
                #    gradBuffer[ix] = grad * 0
                #logger.info('weights')
                #model_ctrl.print_weights()

                # ----Print training details.----
                #logger.info('Outputs')
                #index = []
                #ind = 1
                #while ind < len(state_hist):
                #    index.append(ind-1)
                #    ind += 2000
                #feed_dict = {model_ctrl.state_plh:np.array(state_hist)[index],
                #            model_ctrl.action_plh:np.array(action_hist)[index],
                #            model_ctrl.reward_plh:np.array(reward_hist)[index]}
                #fetch = [model_ctrl.output,
                #         model_ctrl.action,
                #         model_ctrl.reward_plh,
                #         model_ctrl.state_plh,
                #         model_ctrl.logits
                #        ]
                #r = model_ctrl.sess.run(fetch, feed_dict=feed_dict)
                #logger.info('state:\n{}'.format(r[3]))
                #logger.info('output:\n{}'.format(r[0]))
                #logger.info('action: {}'.format(r[1]))
                #logger.info('reward: {}'.format(r[2]))
                #logger.info('logits: {}'.format(r[4]))

            save_model_flag = False
            if config.student_model_name == 'reg':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                loss = model_stud.best_loss
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_loss = loss
                    save_model_flag = True
                    endurance = 0
                else:
                    endurance += 1
                logger.info('best_loss: {}'.format(loss))
                logger.info('lambda1: {}'.format(config.lambda1_stud))
            elif config.student_model_name == 'cls':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                acc = model_stud.best_acc
                loss = model_stud.best_loss
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
                #if ep % config.save_frequency == 0 and ep > 0:
                #    save_model_flag = True
            elif config.student_model_name == 'gan' or\
                config.student_model_name == 'gan_cifar10':
                loss_analyzer_gan(action_hist, reward_hist)
                best_inps = model_stud.best_inception_score
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_best_inps = best_inps
                    save_model_flag = True
                logger.info('best_inps: {}'.format(best_inps))
                logger.info('best_best_inps: {}'.format(best_best_inps))
                logger.info('final_inps_baseline: {}'.\
                            format(model_stud.final_inps_baseline))
            elif config.student_model_name == 'gan_grid':
                loss_analyzer_gan(action_hist, reward_hist)
                hq = model_stud.best_hq
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_hq = hq
                    save_model_flag = True
                logger.info('hq: {}'.format(hq))
                logger.info('best_hq: {}'.format(best_hq))

            logger.info('adv: {}'.format(adv))

            if save_model_flag and save_ctrl:
                    model_ctrl.save_model(ep)
            #if endurance > config.max_endurance_rl:
            #    break

    def test(self, load_ctrl, ckpt_num=None):
        config = self.config
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        model_ctrl.initialize_weights()
        model_stud.initialize_weights()
        model_ctrl.load_model(load_ctrl, ckpt_num=ckpt_num)
        model_stud.reset()

        state = model_stud.get_state()
        for i in range(config.max_training_step):
            action = model_ctrl.sample(state)
            state_new, _, dead = model_stud.response(action, mode='TEST')
            if (i % 10 == 0) and config.student_model_name == 'cls':
                valid_loss = model_stud.previous_valid_loss[-1]
                valid_acc = model_stud.previous_valid_acc[-1]
                train_loss = model_stud.previous_train_loss[-1]
                train_acc = model_stud.previous_train_acc[-1]
                logger.info('Step {}'.format(i))
                logger.info('train_loss: {}, valid_loss: {}'.format(train_loss, valid_loss))
                logger.info('train_acc : {}, valid_acc : {}'.format(train_acc, valid_acc))

            state = state_new
            if dead:
                break
        if config.student_model_name == 'reg':
            loss = model_stud.best_loss
            logger.info('loss: {}'.format(loss))
            return loss
        elif config.student_model_name == 'cls':
            valid_acc = model_stud.best_acc
            test_acc = model_stud.test_acc
            logger.info('valid_acc: {}'.format(valid_acc))
            logger.info('test_acc: {}'.format(test_acc))
            return test_acc
        elif config.student_model_name == 'gan':
            model_stud.load_model(model_stud.task_dir)
            inps_test = model_stud.get_inception_score(500, splits=5)
            model_stud.genrate_images(0)
            logger.info('inps_test: {}'.format(inps_test))
            return inps_test
        elif config.student_model_name == 'gan_grid':
            raise NotImplementedError
        elif config.student_model_name == 'gan_cifar10':
            best_inps = model_stud.best_inception_score
            logger.info('best_inps: {}'.format(best_inps))
            return best_inps
        else:
            raise NotImplementedError

    def baseline(self):
        self.model_stud.initialize_weights()
        self.model_stud.train(save_model=True)
        if self.config.student_model_name == 'gan':
            self.model_stud.load_model(self.model_stud.task_dir)
            inps_baseline = self.model_stud.get_inception_score(500, splits=5)
            self.model_stud.generate_images(0)
            logger.info('inps_baseline: {}'.format(inps_baseline))
        return inps_baseline

    def generate(self, load_stud):
        self.model_stud.initialize_weights()
        self.model_stud.load_model(load_stud)
        self.model_stud.generate_images(0)


if __name__ == '__main__':
    args = parse_args()

    # ----Task name checking.----
    if not args.task_name in ['reg', 'cls', 'gan', 'mnt', 'gan_cifar10']:
        logger.exception('Unexcepted task name')
        exit()

    # ----Mode checking.----
    if not args.task_mode in ['train', 'test', 'baseline', 'generate']:
        logger.exception('Unexcepted mode')
        exit()

    # ----Loading config file.----
    logger.info('Machine name: {}'.format(socket.gethostname()))
    config_path = os.path.join(root_path, 'config/' + args.task_name)
    config = utils.load_config(config_path)

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
            baseline_accs.append(trainer.baseline())
        logger.info(baseline_accs)
    elif args.task_mode == 'generate':
        # ----Generate---- only affect in gan tasks
        logger.info('GENERATE')
        trainer.generate(args.load_stud)


