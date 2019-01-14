import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_path)

def get_cfg():
    return Config()

class Config():
    def __init__(self):
        self.hostname = socket.gethostname()
        # Task definition
        self.task_names = ['mt', 'ner', 'pos'] # The first task is considered as the primary task
        self.task_name_id_dict = {'mt': 0, 'ner': 1, 'pos': 2}

        # Environment & Path
        self.exp_dir = root_path
        self.data_dir = os.path.join(self.exp_dir, 'Data')
        if self.hostname == 'Luna-Desktop':
            self.model_dir = '/media/haowen/autoLoss/saved_models'
        else:
            self.model_dir = '/datasets/BigLearning/haowen/autoLoss/saved_models'

        # Data
        self.train_data_files = [os.path.join(self.data_dir, 'mnt/{}/{}_train.json'.format(task, task))
                                 for task in self.task_names]
        self.valid_data_files = [os.path.join(self.data_dir, 'mnt/{}/{}_valid.json'.format(task, task))
                                 for task in self.task_names]
        self.test_data_files = [os.path.join(self.data_dir, 'mnt/{}/{}_test.json'.format(task, task))
                                for task in self.task_names]

        self.max_seq_length = 60
        self.num_encoder_symbols = 10000
        self.num_decoder_symbols = [10000, 29, 61]

        # Task model
        self.cell_type = 'LSTM'
        self.attention_type = 'luong'
        self.attn_input_feeding = False
        self.use_dropout = True
        self.use_residual = True

        self.embedding_size = 128
        self.encoder_hidden_units = 256
        self.encoder_depth = 2
        self.attn_hidden_units = 256
        self.decoder_hidden_units = 256
        self.decoder_depth = 2

        # Training task model
        self.batch_size = 128
        self.optimizer = 'adam'
        self.lr_task = 0.001
        self.max_gradient_norm = 1.0
        self.keep_prob = 0.7

        self.display_frequency_task = 50
        self.valid_frequency_task = 50
        # Update the controller every short_horizon_len steps
        self.short_horizon_len = 50
        self.save_frequency_task = 500
        self.history_len_task = 5
        self.max_endurance_task = 10
        self.max_training_steps = 100000

        self.max_decode_step = 70
        self.beam_width = 5

        # Controller
        self.buffer_size = 200

        self.dim_input_ctrl = 18
        self.dim_hidden_ctrl = 32
        self.dim_output_ctrl = 3
        self.reward_baseline_decay = 0.8
        self.reward_c = 20000
        # Set an max step reward, in case the improvement baseline is too small
        # and cause huge reward.
        self.reward_max_value = 20
        # TODO check out what this hp do
        self.reward_step_ctrl = 1

        # Training controller
        self.lr_ctrl = 0.001
        self.batch_size_ctrl = 64
        self.gamma_ctrl = 0.9
        self.cliprange_ctrl = 0.2
        self.sync_frequency_ctrl = 1
        self.entropy_bonus_beta_ctrl = 0.00001

        self.total_episodes = 400
        self.rl_method = 'ppo'

        self.epsilon_start_ctrl = 0.1
        self.epsilon_end_ctrl = 0.1
        self.epsilon_decay_steps_ctrl = 50
        self.history_len_task = 10
        self.max_endurance_ctrl = 10

    def print_config(self, logger):
        for key, value in vars(self).items():
            logger.info('{}:: {}'.format(key, value))
