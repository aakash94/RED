import gym
from ReplayBuffer import ReplayBuffer as ReplayBuffer
from AgentDQN import AgentDQN as AgentDQN

import random
import torch
import torch.nn as nn
import torch.optim as optim
from VisdomPlotter import VisdomPlotter
import numpy as np
import os
from collections import deque


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag_reward = "reward"
    tag_loss = "loss"
    tag_ep = "epsilon"

    env = gym.make('CartPole-v0')
    plotter = VisdomPlotter(env_name='CartPole')


    scores = deque(maxlen=100)  # because solution depends on last 100 solution.
    scores.append(int(0))

    BATCH_SIZE = 128
    GAMMA = 0.99  # discount factor
    EPSILON = 1
    LEARN_RATE = 0.01

    UPDATE_TARGET_COUNT = 1

    STATE_N = 4
    ACTION_N = env.action_space.n

    REWARD_THRESHOLD = 0

    OPTIMIZE_COUNT = 8

    NUM_EPISODES = 4096

    qvfa = AgentDQN(STATE_N, ACTION_N).double().to(device)
    q_target = AgentDQN(STATE_N, ACTION_N).double().to(device)
    optimizer = optim.Adam(qvfa.parameters(), lr=LEARN_RATE)

    criterion = nn.MSELoss()
    buffer = ReplayBuffer(10000000)


    def select_action(state, ep=0):
        sample = random.random()
        state = torch.from_numpy(state).to(device)
        if sample < ep:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                qvfa.eval()
                op = qvfa(state)
                values, indices = op.max(0)
                return indices.item()



    def optimize_model(i_episode=0):
        if buffer.__len__() < BATCH_SIZE:
            print("optimizing model Not enough samples in buffer : ", buffer.__len__())
            return

        transitions = buffer.sample(min(BATCH_SIZE, buffer.__len__()))

        state_batch = transitions[buffer.header[0]].values
        state_batch = torch.from_numpy(np.stack(state_batch, axis=0)).to(device)

        action_batch = torch.tensor(transitions[buffer.header[1]].values.tolist()).view(-1, 1).to(device)

        next_state_batch = transitions[buffer.header[2]].values
        next_state_batch = torch.from_numpy(np.stack(next_state_batch, axis=0)).to(device)

        reward_batch = torch.tensor(transitions[buffer.header[3]].values.tolist()).view(-1, 1).to(device)

        done_batch = torch.tensor(transitions[buffer.header[4]].values.tolist()).view(-1, 1).to(device)
        qvfa.train()
        qsa = qvfa(state_batch).gather(1, action_batch)

        with torch.no_grad():
            qvfa.eval()
            q_target.eval()
            next_state_action_values = q_target(next_state_batch)
            max_next_state_values, _indices = next_state_action_values.max(dim=1)
            max_next_state_values = max_next_state_values.view(-1, 1)
            next_state_values = ((max_next_state_values * GAMMA).float() + reward_batch).float() * (1 - done_batch).float()
            target = next_state_values.double()
            qvfa.train()

        #𝛿 = 𝑄(𝑠,𝑎) − (𝑟+𝛾 max𝑎𝑄(𝑠′,𝑎))
        optimizer.zero_grad()
        loss = criterion(qsa, target)
        loss.backward()
        # for param in qvfa.parameters():param.grad.data.clamp_(-1, 1)
        optimizer.step()
        plotter.plot_line('mse loss', 'loss', 'Loss every optimization', i_episode, loss.item())


    def is_solved(value, threshold=195):

        scores.append(int(value))
        score = sum(scores) / 100

        if score >= threshold:
            print("SOLVED")
            return True

        return False


    def rbed(current_reward, eps=EPSILON, highest_ep=1.0, lowest_ep=0.01, target_reward=200,
             target_increment=1,reward_threshold=0):

        steps_to_move_in = target_reward
        quanta = (highest_ep - lowest_ep) / steps_to_move_in
        if current_reward > reward_threshold:
            reward_threshold += target_increment
            eps -= quanta

        return max(eps, lowest_ep), reward_threshold

    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        # 1. Avoid moving targets
        if i_episode % UPDATE_TARGET_COUNT == 0:
            q_target.load_state_dict(qvfa.state_dict())

        while not done:
            env.render()
            action = select_action(state, ep=EPSILON)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            doneV = 0
            if done:
                reward = -reward
                doneV = 1

            buffer.insert(state, action, next_state, reward, doneV)
            state = next_state


        plotter.plot_line('epsilon', 'rbed_epsilon', 'RBED Epsilon', i_episode, EPSILON)
        plotter.plot_line('total reward', 'rbed_reward', 'Reward', i_episode, total_reward)


        if is_solved(total_reward):
            print('solved at episode ', i_episode)
            break

        for _ in range(OPTIMIZE_COUNT):
            optimize_model(i_episode)

        EPSILON, REWARD_THRESHOLD = rbed(total_reward, eps=EPSILON, reward_threshold=REWARD_THRESHOLD)

    print('Complete')
    env.render()
    env.close()





if __name__ == '__main__':
    main()
