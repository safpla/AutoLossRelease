import tensorflow as tf
import gym
import cv2
import numpy as np
from ppo import MlpPPO
from config import *

NUM_OF_GAMES = 100

env = gym.make(args.environment)
action_space = env.action_space
observation_space = env.observation_space
action_bound = [env.action_space.low, env.action_space.high]
global_episodes, global_update_counter = 0, 0
x_t = env.reset()


play_network = MlpPPO(action_space, observation_space,'Chief', args)

sess = tf.InteractiveSession()

saver = tf.train.Saver(max_to_keep=5)
play_network.load_model(sess, saver)

score = 0

game_no = 1

total_sore = []
while True:

    env.render()

    action = play_network.choose_action(x_t, sess)
    x_t, r_t, terminal, info = env.step(action)
    score += r_t

    if terminal:
        print('Game No : '+ str(game_no)+ ', Score : ', score)
        x_t = env.reset()
        game_no += 1
        total_sore.append(score)
        score = 0
        if game_no == NUM_OF_GAMES + 1:
            break


avarage_score = np.vstack(total_sore).mean()
print('Avarage Score in 100 Games : ' + str(avarage_score))



