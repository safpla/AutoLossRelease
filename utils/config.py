from configparser import ConfigParser, ExtendedInterpolation
import json
import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
import utils
logger = utils.get_logger()

class Parser(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), '{} not exists.'.format(config_path)
        self.config = ConfigParser(
            delimiters='=',
            interpolation=ExtendedInterpolation())
        self.config.read(config_path)
        try:
            self.lambda1_stud = self.config.getfloat('stud', 'lambda1_stud')
        except:
            logger.warning('lambda1_stud not found in config file')
            self.lambda1_stud = 0
        try:
            self.lambda2_stud = self.config.getfloat('stud', 'lambda2_stud')
        except:
            logger.warning('lambda2_stud not found in config file')
            self.lambda2_stud = 0

    @property
    def random_seed(self):
        return self.config.getint('data', 'random_seed')

    def print_config(self):
        for key_sec, sec in self.config.items():
            logger.info('[{}]'.format(key_sec))
            for key, value in sec.items():
                logger.info('{}:: {}'.format(key, value))

    @property
    def lr_decay_stud(self):
        return self.config.getfloat('stud', 'lr_decay_stud')

    @property
    def lr_rl(self):
        return self.config.getfloat('rl', 'lr_rl')

    @property
    def lr_decay_rl(self):
        return self.config.getfloat('rl', 'lr_decay_rl')

    @property
    def num_pre_loss(self):
        return self.config.getint('rl', 'num_pre_loss')

    @property
    def dim_state_rl(self):
        return self.config.getint('rl', 'dim_state_rl')

    @property
    def dim_hidden_rl(self):
        return self.config.getint('rl', 'dim_hidden_rl')

    @property
    def dim_action_rl(self):
        return self.config.getint('rl', 'dim_action_rl')

    @property
    def reward_c(self):
        return self.config.getfloat('rl', 'reward_c')

    @property
    def reward_step_rl(self):
        return self.config.getfloat('rl', 'reward_step_rl')

    @property
    def explore_rate_decay_rl(self):
        return self.config.getint('rl', 'explore_rate_decay_rl')

    @property
    def explore_rate_rl(self):
        return self.config.getfloat('rl', 'explore_rate_rl')

    @property
    def total_episodes(self):
        return self.config.getint('rl', 'total_episodes')

    @property
    def max_training_step(self):
        return self.config.getint('stud', 'max_training_step')

    @property
    def max_ctrl_step(self):
        return self.config.getint('rl', 'max_ctrl_step')

    @property
    def update_frequency(self):
        return self.config.getint('rl', 'update_frequency')

    @property
    def save_frequency(self):
        return self.config.getint('rl', 'save_frequency')

    @property
    def exp_dir(self):
        if socket.gethostname() == 'Luna-Desktop':
            return os.path.expanduer(self.config.get('env', 'exp_dir1'))
        else:
            return os.path.expanduser(self.config.get('env', 'exp_dir'))

    @property
    def data_dir(self):
        if socket.gethostname() == 'Luna-Desktop':
            return os.path.expanduser(self.config.get('env', 'data_dir1'))
        else:
            return os.path.expanduser(self.config.get('env', 'data_dir'))

    @property
    def model_dir(self):
        if socket.gethostname() == 'Luna-Desktop':
            return os.path.expanduser(self.config.get('env', 'model_dir1'))
        else:
            return os.path.expanduser(self.config.get('env', 'model_dir'))

    @property
    def save_images_dir(self):
        return os.path.expanduser(self.config.get('env', 'save_images_dir'))

    @property
    def student_model_name(self):
        return self.config.get('stud', 'student_model_name')

    @property
    def controller_model_name(self):
        return self.config.get('rl', 'controller_model_name')

    @property
    def train_data_file(self):
        train_data_file = self.config.get('data', 'train_data_file')
        return os.path.join(self.data_dir, train_data_file)

    @property
    def valid_data_file(self):
        valid_data_file = self.config.get('data', 'valid_data_file')
        return os.path.join(self.data_dir, valid_data_file)

    @property
    def train_stud_data_file(self):
        train_stud_data_file = self.config.get('data', 'train_stud_data_file')
        return os.path.join(self.data_dir, train_stud_data_file)

    @property
    def test_data_file(self):
        test_data_file = self.config.get('data', 'test_data_file')
        return os.path.join(self.data_dir, test_data_file)

    @property
    def num_sample_train(self):
        return self.config.getint('data', 'num_sample_train')

    @property
    def num_sample_valid(self):
        return self.config.getint('data', 'num_sample_valid')

    @property
    def num_sample_test(self):
        return self.config.getint('data', 'num_sample_test')

    @property
    def num_sample_train_stud(self):
        return self.config.getint('data', 'num_sample_train_stud')

    @property
    def mean_noise(self):
        return self.config.getfloat('data', 'mean_noise')

    @property
    def var_noise(self):
        return self.config.getfloat('data', 'var_noise')

    @property
    def batch_size(self):
        return self.config.getint('stud', 'batch_size')

    @property
    def dim_input_stud(self):
        return self.config.getint('stud', 'dim_input_stud')

    @property
    def dim_hidden_stud(self):
        return self.config.getint('stud', 'dim_hidden_stud')

    @property
    def dim_output_stud(self):
        return self.config.getint('stud', 'dim_output_stud')

    @property
    def lr_stud(self):
        return self.config.getfloat('stud', 'lr_stud')

    @property
    def beta1(self):
        return self.config.getfloat('stud', 'beta1')

    @property
    def beta2(self):
        return self.config.getfloat('stud', 'beta2')

    @property
    def valid_frequency_stud(self):
        return self.config.getint('stud', 'valid_frequency_stud')

    @property
    def max_endurance_stud(self):
        return self.config.getint('stud', 'max_endurance_stud')

    @property
    def print_frequency_stud(self):
        return self.config.getint('stud', 'print_frequency_stud')

    @property
    def max_endurance_rl(self):
        return self.config.getint('rl', 'max_endurance_rl')

    @property
    def inps_baseline_decay(self):
        return self.config.getfloat('rl', 'inps_baseline_decay')

    @property
    def reward_baseline_decay(self):
        return self.config.getfloat('rl', 'reward_baseline_decay')

    @property
    def reward_max_value(self):
        return self.config.getfloat('rl', 'reward_max_value')

    @property
    def reward_min_value(self):
        return self.config.getfloat('rl', 'reward_min_value')

    @property
    def logit_clipping_c(self):
        return self.config.getfloat('rl', 'logit_clipping_c')

    @property
    def dim_z(self):
        return self.config.getint('gan', 'dim_z')

    @property
    def dim_x(self):
        return self.config.getint('gan', 'dim_x')

    @property
    def dim_c(self):
        return self.config.getint('gan', 'dim_c')

    @property
    def n_hidden_disc(self):
        return self.config.getint('gan', 'n_hidden_disc')

    @property
    def n_hidden_gen(self):
        return self.config.getint('gan', 'n_hidden_gen')

    @property
    def disc_iters(self):
        return self.config.getint('gan', 'disc_iters')

    @property
    def gen_iters(self):
        return self.config.getint('gan', 'gen_iters')

    @property
    def state_decay(self):
        return self.config.getfloat('rl', 'state_decay')

    @property
    def metric_decay(self):
        return self.config.getfloat('rl', 'metric_decay')

    @property
    def stop_strategy_stud(self):
        return self.config.get('stud', 'stop_strategy_stud')

    @property
    def inps_threshold(self):
        return self.config.getfloat('gan', 'inps_threshold')

    @property
    def inps_batches(self):
        return self.config.getint('gan', 'inps_batches')

    @property
    def inps_splits(self):
        return self.config.getint('gan', 'inps_splits')

    @property
    def optimizer_ctrl(self):
        return self.config.get('rl', 'optimizer_ctrl')


def load_config(cfg_dir):
    '''
        Raises:
            FileNotFoundError if 'cfg.py' doesn't exist in cfg_dir
    '''
    if not os.path.isfile(os.path.join(cfg_dir, 'cfg.py')):
        raise ImportError('cfg.py not found in {}'.format(cfg_dir))
    import sys
    sys.path.insert(0, cfg_dir)
    from cfg import get_cfg
    cfg = get_cfg()
    # cleanup
    try:
        del sys.modules['cfg']
    except:
        pass
    sys.path.remove(cfg_dir)

    return cfg

def override_config(config, args):
    if args.lambda_task:
        config.lambda_task = args.lambda_task

    if args.disc_iters:
        config.disc_iters = args.disc_iters

    if args.gen_iters:
        config.gen_iters = args.gen_iters





if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = Parser(config_path)
    config.print_config()
