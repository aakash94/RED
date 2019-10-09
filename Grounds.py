import gym
from .ReplayBuffer import ReplayBuffer
from .AgentDQN import AgentDQN
from stable_baselines import DQN

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE = 'VANILA'
#MODE = 'RBED'

writer = SummaryWriter(comment=MODE)
tag_reward = "reward"
tag_loss = "loss"
tag_ep = "epsilon"

discrete_control = ['CartPole-v0', 'MountainCar-v0']
continuous_control = ['','']
print('hello world')
env = gym.make(discrete_control[0])


BATCH_SIZE = 32
GAMMA      = 0.9 # discount factor
EPSILON    = 1
EPSILON_DECAY    = 0.99
LEARN_RATE = 0.001

CHECK_EVERY = 100
OPTIMIZE_EVERY = 1

STATE_N  = 4
ACTION_N = env.action_space.n

OPTIMIZE_COUNT = 1

NUM_EPISODES = 10000


MINREWARD = 30


qvfa = AgentDQN(STATE_N, ACTION_N).double().to(device)
optimizer = optim.Adam(qvfa.parameters(), lr = LEARN_RATE)

criterion = nn.MSELoss()
buffer = ReplayBuffer(1000000)


def select_action(state, ep=0):
    sample = random.random()
    state = torch.from_numpy(state).to(device)
    if sample < ep:
        return env.action_space.sample()
    else:
        with torch.no_grad():
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

    qsa = qvfa(state_batch).gather(1, action_batch)

    with torch.no_grad():
        qvfa.eval()
        next_state_action_values = qvfa(next_state_batch)
        max_next_state_values, _indices = next_state_action_values.max(dim=1)
        max_next_state_values = max_next_state_values.view(-1, 1)
        next_state_values = ((max_next_state_values * GAMMA).float() + reward_batch).float() * (1 - done_batch).float()
        target = next_state_values.double()
        qvfa.train()

    # 𝛿=𝑄(𝑠,𝑎)−(𝑟+𝛾max𝑎𝑄(𝑠′,𝑎))
    optimizer.zero_grad()
    loss = criterion(qsa, target)
    loss.backward()
    # for param in qvfa.parameters():param.grad.data.clamp_(-1, 1)
    optimizer.step()
    writer.add_scalar(tag_loss, loss.item(), i_episode)

def standard_decay (episode, highest_reward=0, eps=EPSILON):
    return  eps

def rbed(episode, highest_reward=0, eps=EPSILON):
    return eps

def decay_epsilon(episode, highest_reward=0, eps=EPSILON):
    if MODE == 'RBED':
        eps = rbed(episode, highest_reward, eps)
    else:
        eps = standard_decay(episode, highest_reward, eps)

    eps = eps if eps < EPSILON else EPSILON
    return eps

for i_episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render(mode='rgb_array')
        action = select_action(state, ep=EPSILON)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            reward = -reward
        buffer.insert(state, action, next_state, reward, done)
        state = next_state

    writer.add_scalar(tag_reward, total_reward, i_episode)
    writer.add_scalar(tag_ep, EPSILON, i_episode)
    for _ in range(OPTIMIZE_COUNT):
        optimize_model(i_episode)

    EPSILON = decay_epsilon(i_episode, total_reward, EPSILON)
    # if EPSILON > 0.2 and i_episode > 32 and total_reward > MINREWARD:
    #     EPSILON -= 0.1
    #     MINREWARD += 20

print('Complete')
env.render()
env.close()