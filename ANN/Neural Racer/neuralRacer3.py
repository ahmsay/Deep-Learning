import numpy as np
import random
from Game import Game

env = Game()
action_size = 2
state_size = 4
qtable = np.zeros((state_size, action_size))

total_episodes = 100
learning_rate = 1
max_steps = 100
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)
        
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.randomChoice()
            
        new_state, reward, done = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        total_rewards += reward
        state = new_state

        if done == True:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)
    

env.reset()

for episode in range(10):
    state = env.reset()
    step = 0
    done = False

    for step in range(100):
        action = np.argmax(qtable[state,:])
        new_state, reward, done = env.step(action)
        
        if done:
            break
        state = new_state
    print("Generation", episode, ":", step)