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
        self.data_dir = os.path.join(self.exp_dir, 'Data')
        if self.hostname == 'Luna-Desktop':
            self.model_dir = '/media/haowen/autoLoss/saved_models'
        else:
            self.model_dir = '/datasets/BigLearning/haowen/autoLoss/saved_models'

        # Data
        self.train_ctrl_data_file = 'cls/train_ctrl.npy'
        self.valid_ctrl_data_file = 'cls/valid_ctrl.npy'
        self.train_task_data_file = 'cls/train_task.npy'
        self.valid_task_data_file = 'cls/valid_task.npy'
        self.test_data_file = 'cls/test.npy'
        self.num_sample_train_ctrl = 200
        self.num_sample_valid_ctrl = 1000
        self.num_sample_train_task = 200
        self.num_sample_valid_task = 1000
        self.num_sample_test = 1000
        self.mean_noise = 0
        self.var_noise = 4
        # 1 for training data, others for transfer learning
        self.random_seed = 1

        # Task model
        self.dim_input_task = 32
        self.dim_hidden_task = 32
        self.dim_output_task = 2
        self.lambda_task = 0.08

        # Training task model
        self.batch_size = 200
        self.lr_task = 0.001
        self.valid_frequency_task = 10
        self.print_frequency_task = 50
        # options for `stop_strategy_task` are: exceeding_endurance,
        # exceeding_total_steps
        #self.stop_strategy_task = 'exceeding_total_steps'
        self.stop_strategy_task = 'exceeding_endurance'
        self.max_endurance_task = 20
        self.max_training_step = 2000

        # Controller
        self.controller_model_name = '2layer_logits_clipping'
        #self.controller_model_name = 'linear_logits_clipping'
        # "How many recent training steps will be recorded"
        self.num_pre_loss = 2

        self.dim_input_ctrl = 6
        self.dim_hidden_ctrl = 16
        self.dim_output_ctrl = 2
        self.reward_baseline_decay = 0.9
        self.reward_c = 1000
        # Set an max step reward, in case the improvement baseline is too small
        # and cause huge reward.
        self.reward_max_value = 5
        # TODO: 0.1
        self.reward_step_ctrl = 0.1
        self.logit_clipping_c = 2

        # Training controller
        self.lr_ctrl = 0.001
        self.total_episodes = 40000
        self.update_frequency_ctrl = 1
        self.save_frequency_ctrl = 50
        self.max_endurance_ctrl = 1000
        self.rl_method = 'reinforce'
        self.epsilon_start_ctrl = 0.5
        self.epsilon_end_ctrl = 0.1
        self.epsilon_decay_steps_ctrl = 1

    def print_config(self, logger):
        for key, value in vars(self).items():
            logger.info('{}:: {}'.format(key, value))
