import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--environment', type=str, default='Pendulum-v0')
parser.add_argument('--policy', type=str, default='MlpPolicy')
parser.add_argument('--checkpoint_dir', type=str, default='./save_model')
parser.add_argument('--summary_dir', type=str, default='./summary_log')
parser.add_argument('--a_learning_rate', type=float, default=0.0001)
parser.add_argument('--c_learning_rate', type=float, default=0.0002)
parser.add_argument('--cliprange', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--training_step', type=int, default=10)
parser.add_argument('--gamma', type=float, default= 0.9)



args = parser.parse_args()
