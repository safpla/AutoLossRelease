import numpy as np
import utils

logger = utils.get_logger()

def get_reward(losses):
    zero = 0
    pos = 0
    neg = 0
    for l in losses:
        if abs(l) < 1e-5:
            zero += 1
        elif l > 0:
            pos += 1
        else:
            neg += 1
    return zero, pos, neg

def loss_analyzer_toy(transitions):
    actions = [trans['action'] for trans in transitions]
    valid_losses = [trans['valid_loss'] for trans in transitions]
    train_losses = [trans['train_loss'] for trans in transitions]
    rewards = [trans['reward'] for trans in transitions]

    total_steps = len(actions)
    logger.info('total_steps: {}'.format(total_steps))

    # ----Prior of each action.----
    action_sum = np.sum(np.array(actions), axis=0) / total_steps
    logger.info('p_a: {}'.format(action_sum))

    loss_mse = []
    for idx, a in enumerate(actions):
        if idx == 0:
            continue
        if a[0] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_mse.append(loss_diff)

    loss_l1 = []
    for idx, a in enumerate(actions):
        if idx == 0:
            continue
        if a[1] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_l1.append(loss_diff)

    # ----Mean and Var of loss improvement of each action.----
    #logger.info('loss_mse_mean: {}, var: {}'.format(
    #    np.mean(np.array(loss_mse)), np.var(np.array(loss_mse))))
    #logger.info('loss_l1_mean: {}, var: {}'.format(
    #    np.mean(np.array(loss_l1)), np.var(np.array(loss_l1))))

    # ----Step reward distribution.----
    #zero, pos, neg = get_reward(loss_mse)
    #logger.info('MSE:: zero: {}, pos: {}, neg: {}'.format(zero, pos, neg))

    #zero, pos, neg = get_reward(loss_l1)
    #logger.info('L1 :: zero: {}, pos: {}, neg: {}'.format(zero, pos, neg))

    # ----Distribution of each action over time.----
    win = 100
    loss_imp_mse_trace = []
    loss_imp_l1_trace = []
    mse_dis_trace = []
    l1_dis_trace = []
    for i in range(min(80, int(total_steps / win))):
        start = i * win
        stop = (i + 1) * win
        action = actions[start:stop]
        valid_loss = valid_losses[start:stop]
        loss_mse = []
        loss_l1 = []
        for idx, a in enumerate(action):
            if idx == 0:
                continue
            if a[0] == 1:
                loss_diff = valid_loss[idx - 1] - valid_loss[idx]
                loss_mse.append(loss_diff)
            elif a[1] == 1:
                loss_diff = valid_loss[idx - 1] - valid_loss[idx]
                loss_l1.append(loss_diff)

        loss_imp_mse_trace.append(np.mean(np.array(loss_mse)))
        loss_imp_l1_trace.append(np.mean(np.array(loss_l1)))
        mse_dis_trace.append(len(loss_mse))
        l1_dis_trace.append(len(loss_l1))

    logger.info('Trace of actions distribution')
    logger.info('mse: {}'.format(mse_dis_trace))
    logger.info('l1: {}'.format(l1_dis_trace))

    #logger.info('Trace of loss improvement:')
    #logger.info('mse: {}'.format(loss_imp_mse_trace))
    #logger.info('l1: {}'.format(loss_imp_l1_trace))

    # ----Distribution of reward.----
    reward_mse_sum_trace = []
    reward_mse_mean_trace = []
    reward_l1_sum_trace = []
    reward_l1_mean_trace = []
    for i in range(min(80, int(total_steps / win))):
        start = i * win
        stop = (i + 1) * win
        action = actions[start:stop]
        reward = rewards[start:stop]
        reward_mse = []
        reward_l1 = []
        for idx, a in enumerate(action):
            if a[0] == 1:
                reward_mse.append(reward[idx])
            elif a[1] == 1:
                reward_l1.append(reward[idx])
        reward_mse_sum_trace.append(np.sum(np.array(reward_mse)))
        reward_l1_sum_trace.append(np.sum(np.array(reward_l1)))
        reward_mse_mean_trace.append(np.mean(np.array(reward_mse)))
        reward_l1_mean_trace.append(np.mean(np.array(reward_l1)))
    #logger.info('Trace of rewards sum')
    #logger.info('mse: {}'.format(reward_mse_sum_trace))
    #logger.info('l1: {}'.format(reward_l1_sum_trace))

    #logger.info('Trace of rewards mean')
    #logger.info('mse: {}'.format(reward_mse_mean_trace))
    #logger.info('l1: {}'.format(reward_l1_mean_trace))

def loss_analyzer_gan(actions, rewards):
    total_steps = len(actions)
    logger.info('total_steps: {}'.format(total_steps))

    # ----Prior of each action.----
    action_sum = np.sum(np.array(actions), axis=0) / total_steps
    logger.info('p_a: {}'.format(action_sum))

    # ----Distribution of each action over time.----
    win = 100
    update_gen_trace = []
    update_disc_trace = []
    for i in range(int(total_steps / win)):
        start = i * win
        stop = (i + 1) * win
        action = actions[start:stop]
        sum_action = np.sum(np.array(action), 0)
        update_gen_trace.append(sum_action[0])
        update_disc_trace.append(sum_action[1])

    logger.info('Trace of actions distribution')
    logger.info('gen: {}'.format(update_gen_trace))
    logger.info('disc: {}'.format(update_disc_trace))
