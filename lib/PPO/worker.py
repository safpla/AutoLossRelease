import tensorflow as tf
from ppo import *
import random


class Worker(object):

    def __init__(self, name, env, summary_writer, global_episodes, args):
        self.scope = name
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.summary_writer = summary_writer
        self.global_episodes = global_episodes
        self.increse_global_episodes = self.global_episodes.assign_add(1)
        if args.policy == 'LstmPolicy':
            self.ppo = LstmPPO(self.action_space, self.observation_space,self.scope, args)
        else:
            self.ppo = MlpPPO(self.action_space, self.observation_space,self.scope, args)

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.training_step = args.training_step
        self.train_op_actor = tf.train.AdamOptimizer(args.a_learning_rate).minimize(self.ppo.actor_loss)
        self.train_op_critic = tf.train.AdamOptimizer(args.c_learning_rate).minimize(self.ppo.critic_loss)



    def process(self, sess, saver):

        x_t = self.env.reset()
        x_t1 = 0
        total_rewards = 0
        episode_length = 0
        global_episodes = 0
        self.num_training = 0

        while True:
            states_buf = []
            actions_buf = []
            rewards_buf = []

            for i in range(0, self.batch_size):

                global_episodes = sess.run(self.increse_global_episodes)

                self.env.render()

                action = self.ppo.choose_action(x_t, sess)

                x_t1, r_t, self.terminal, info = self.env.step(action)
                total_rewards += r_t
                episode_length += 1
                states_buf.append(x_t)
                actions_buf.append(action)
                #rewards_buf.append((r_t+8)/8)
                #rewards_buf.append(r_t)
                reward = max(min(r_t, 1), -1)
                rewards_buf.append(reward)
                x_t = x_t1
                if self.terminal:
                    print('ID :' + self.scope + ', global episode :' + str(
                    global_episodes)+ ', episode length :' + str(episode_length)+ ', total reward :' + str(total_rewards))
                    x_t = self.env.reset()
                    summary = tf.Summary()
                    summary.value.add(tag='Rewards/Total_Rewards', simple_value=float(total_rewards))
                    summary.value.add(tag='Rewards/Episode_Length', simple_value=float(episode_length))
                    self.summary_writer.add_summary(summary, global_episodes)
                    self.summary_writer.flush()
                    total_rewards = 0
                    episode_length = 0
                    break



            bootstrap_value = self.ppo.get_v(x_t1, sess)  #[0, 0]

            if states_buf:
                discounted_r = []
                v_s_ = bootstrap_value
                for r in rewards_buf[::-1]:
                    v_s_ = r + self.gamma * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(states_buf), np.vstack(actions_buf), np.array(discounted_r)[:, np.newaxis]
                self.train(sess, saver, bs, ba, br)





    def summary_log(self, sess, feed_dict_actor, feed_dict_critic, global_episodes):

        actor_loss = sess.run(self.ppo.actor_loss, feed_dict=feed_dict_actor)
        critic_loss = sess.run(self.ppo.critic_loss, feed_dict=feed_dict_critic)


        summary = tf.Summary()
        summary.value.add(tag='Losses/Actor_Loss', simple_value=float(actor_loss))
        summary.value.add(tag='Losses/Critic_Loss', simple_value=float(critic_loss))
        self.summary_writer.add_summary(summary, global_episodes)
        self.summary_writer.flush()


    def train(self,sess,saver, s, a, r):
        sess.run(self.ppo.syn_old_pi)
        global_episodes = sess.run(self.global_episodes)
        self.num_training += 1

        adv = sess.run(self.ppo.adv, {self.ppo.s: s, self.ppo.y: r})
        feed_dict_actor = {}
        feed_dict_actor[self.ppo.s] = s
        feed_dict_actor[self.ppo.a] = a
        feed_dict_actor[self.ppo.advantage] = adv
        feed_dict_critic = {}
        feed_dict_critic[self.ppo.s] = s
        feed_dict_critic[self.ppo.y] = r

        [sess.run(self.train_op_actor, feed_dict=feed_dict_actor) for _ in range(self.training_step)]
        [sess.run(self.train_op_critic, feed_dict=feed_dict_critic) for _ in range(self.training_step)]

        self.summary_log(sess, feed_dict_actor, feed_dict_critic, global_episodes)

        if self.num_training % 500 == 0:
            self.ppo.save_model(sess, saver, global_episodes)







