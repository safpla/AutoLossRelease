import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_path)

def get_cfg():
    return Config()

class Config():
    def __init__(self):
        self.hostname = socket.gethostname()
        # Environment & Path
        self.exp_dir = root_path

        # Set the path to MNIST dataset below
        self.data_dir = 'data/gan_cifar10/cifar-10-batches-py'
        self.save_images_dir = 'data/gan_cifar10/saved_images'

        self.model_dir = 'ckpts'
        self.pretrained_mnist_checkpoint_dir = os.path.join(self.model_dir, 'mnist_classification')

        # Data

        # Task model
        self.dim_z = 128
        self.dim_x = 3072
        self.dim_c = 64
        self.disc_iters = 1
        self.gen_iters = 1
        self.inps_batches = 50
        self.inps_splits = 1

        # Training task model
        self.batch_size = 256
        self.lr_task = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.valid_frequency_task = 500
        self.print_frequency_task = 500
        self.stop_strategy_task = 'exceeding_endurance'
        self.max_endurance_task = 50
        self.max_training_step = 400000

        # Controller
        self.controller_model_name = '2layer_logits_clipping'
        # "How many recent training steps will be recorded"
        self.num_pre_loss = 2

        self.dim_input_ctrl = 4
        self.dim_hidden_ctrl = 16
        self.dim_output_ctrl = 2

        self.reward_baseline_decay = 0.9
        self.reward_c = 10
        # Set an max step reward, in case the improvement baseline is too small
        # and cause huge reward.
        self.reward_max_value = 20
        self.reward_min_value = 1
        self.reward_step_ctrl = 0.1
        self.logit_clipping_c = 1

        # Training controller
        self.lr_ctrl = 0.001
        self.total_episodes = 400
        self.update_frequency_ctrl = 1
        self.print_frequency_ctrl = 100
        self.save_frequency_ctrl = 100
        self.max_endurance_ctrl = 100
        self.rl_method = 'reinforce'
        self.state_decay = 0.9
        self.metric_decay = 0.8

    def print_config(self, logger):
        for key, value in vars(self).items():
            logger.info('{}:: {}'.format(key, value))
