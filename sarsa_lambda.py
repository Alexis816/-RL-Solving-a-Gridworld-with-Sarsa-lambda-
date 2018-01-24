import gym
import numpy as np

from numpy.random import random, choice
import matplotlib.pyplot as plt

def epsilon_greedy(state, Q, epsilon):

  actions_values = Q[state,:]
  greedy_action = np.argmax(actions_values)
  explore = (random() < epsilon)

  if explore:
    return choice([a for a in range(len(actions_values))])
  else:
    return greedy_action

def moving_average(arr, M):
  result = np.zeros(len(arr)-M+1)

  for i in range(len(result)):
    result[i] = np.mean(arr[i:i+M])

  return result

env = gym.make("FrozenLake-v0")

# parameters for TD(lambda)
episodes = 10000
gamma = 1.0
alpha = 0.1
epsilon = 0.05
eligibility_decay = 0.5

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

all_returns = []

for episode in range(episodes):

  epsilon = 1 - episode / episodes;

  state = env.reset()
  action = epsilon_greedy(state, Q, epsilon)

  R = [None]
  E = np.zeros((n_states, n_actions))

  while True:

    E = eligibility_decay * gamma * E
    E[state, action] += 1

    new_state, reward, done, info = env.step(action)
    new_action = epsilon_greedy(new_state, Q, epsilon)

    R.append(reward)

    delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
    Q = Q + alpha * delta * E 

    state, action = new_state, new_action

    if done:
       break

  T = len(R)
  G = np.zeros(T)

  # t = T-2, T-3, ..., 0
  t = T - 2

  while t >= 0:
    G[t] = R[t+1] + gamma * G[t+1]
    t = t - 1

  all_returns = all_returns + list(G)

moving_average_returns = 100 * moving_average(all_returns, M=100)
plt.plot(moving_average_returns)
plt.show()