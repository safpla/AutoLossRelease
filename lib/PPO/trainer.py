import tensorflow as tf
import gym
from ppo import *
from worker import Worker
from config import *



tf.reset_default_graph()
global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

summary_writer = tf.summary.FileWriter(args.summary_dir + '/' + args.environment + '/' + args.policy)

env = gym.make(args.environment)
chief =  Worker('Chief', env, summary_writer, global_episodes, args)

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    chief.ppo.load_model(sess, saver)
    chief.process(sess, saver)
