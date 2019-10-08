import gym
from stable_baselines import DQN


discrete_control = ['CartPole-v0', 'MountainCar-v0']
continuous_control = ['','']
print('hello world')
env = gym.make(discrete_control[0])