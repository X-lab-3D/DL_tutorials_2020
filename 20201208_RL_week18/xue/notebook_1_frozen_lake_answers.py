#!/usr/bin/env python
# coding: utf-8

# # Building a Deep Q-Network (DQN) for the frozen lake problem 

# In this notebook we will be using the "Frozen lakle" environment provided by OpenAI GYM. 
# ![frozen lake](./images/frozen_lake.jpeg)

# ## Description

# Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.  
# 
# The surface is described using a grid like the following

# SFFF       (S: starting point, safe)  
# FHFH       (F: frozen surface, safe)  
# FFFH       (H: hole, fall to your doom)  
# HFFG       (G: goal, where the frisbee is located)  

# **Actions space, observation space**  
# Recall that our environment has an action space and an observation space. For this basic version of the Frozen Lake game, an observation is a discrete integer value from 0 to 15. This represents the location our character is on. Then the action space is an integer from 0 to 3, for each of the four directions we can move. So our “Q-table” will be an array with 16 rows and 4 columns.
# 
# **Episode termination and reward**  
# The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

# # Installing the correct dependencies

# In[1]:


get_ipython().system('pip3 install numpy')
get_ipython().system('pip3 install gym')
get_ipython().system('pip3 install tqdm')


# In[2]:


import numpy
import gym
import time
from IPython.display import clear_output
from numpy.random import randint


# # The behavior of a random agent
# Let's visualise the behaviour of a random agent first. For now we want to exclude randomness so we set `is_slippery=False`. 

# In[3]:


env = gym.make('FrozenLake-v0', is_slippery=False)
obs = env.reset()

obs, done, rew = env.reset(), False, 0
while (done != True) :
    A =  randint(0,env.action_space.n,(1,))
    obs, reward, done, info = env.step(A.item())
    rew += reward
    env.render()
    print(f'Action: {A.item()}')
    print(f'State: {env.s}')


# Play around with the code block a few times! You can deduce that the possible states are numbered like this:  
# 0 1 2 3  
# 4 5 6 7  
# 8 9 10 11  
# 12 13 14 15  
# 
# And actions are:  
# Up: 3  
# Down: 1  
# Left: 0  
# Right: 2  
# 
# Furthermore, the current state can be retrieved using `env.s`. And `env` has a number of properties which can be retrieved with `dir(env)`

# In[4]:


dir(env)


# # Initializing the Q-table
# Recall that our environment has an action space and an observation space. For this basic version of the Frozen Lake game, an observation is a discrete integer value from 0 to 15. This represents the location our character is on. Then the action space is an integer from 0 to 3, for each of the four directions we can move. So our “Q-table” will be an array with 16 rows and 4 columns.

# In[5]:


# Initialize the Q-table below
# Try not to hard code this, but allow for an environment that can change is size
def init_Q_table(env):
    Q = numpy.zeros((env.observation_space.n, env.action_space.n))
    return Q


# How does this help us choose our move? Well, each cell in this table has a score. This score tells us how good a particular move is for a particular observation state. So we could define a `choose_action` function in a simple way. This will look at the different values in the row for this observation, and choose the highest index. So if the “0” value in this row is the highest, we’ll return 0, indicating we should move left. If the second value is highest, we’ll return 1, indicating a move down.
# 
# But we don’t want to choose our moves deterministically! Our Q-Table starts out in the “untrained” state. And we need to actually find the goal at least once to start back-propagating rewards into our maze. This means we need to build some kind of exploration into our system. So each turn, we can make a random move with probability epsilon.

# In[6]:


# Finish the function below, so that it takes a random move with probability epsilon, 
# and otherwise choses the action with the highest Q-value
def choose_action(state, epsilon):
    action = 0
    if numpy.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = numpy.argmax(Q[state, :])
    return action


# # Updating the Q-Table
# 
# Now, we also want to be able to update our table. To do this, we’ll write a function that follows the Q-learning rule. It will take two states, the reward for the second observation, and the action we took to get there. Finish the function `learn` below and implement the Q-learning algorithm from the slides

# In[7]:


def learn(state1, state2, reward, action):
    prediction = Q[state1, action]
    target = reward + gamma * numpy.max(Q[state2, :])
    Q[state1, action] = Q[state1, action] + learning_rate * (target - prediction)


# # Playing the Game
# 
# Playing the game now is straightforward, following the examples we’ve done before. We’ll have a certain number of episodes. Within each episode, we make our move, and use the reward to “learn” for our Q-table.  
# 
# Finish the training loop below. Implement the following features:
# 1. Reset the environment every episode
# 2. Multiply the the `epsilon` with `decay_rate` every 100 episodes, but it should remain larger than `min_epsilon`

# In[10]:


env = gym.make('FrozenLake-v0', is_slippery=False)
Q = init_Q_table(env)

epsilon = 0.9
min_epsilon = 0.01
decay_rate = 0.9
total_episodes = 10000
max_steps = 100
learning_rate = 0.81
gamma = 0.96

for episode in range(total_episodes):
    obs = env.reset()
    t = 0
    
    if episode % 100 == 99:
        epsilon *= decay_rate
        epsilon = max(epsilon, min_epsilon)  
        
    while t < max_steps:
        action = choose_action(obs, epsilon)
        obs2, reward, done, info = env.step(action)
        learn(obs, obs2, reward, action)
        obs = obs2
        t += 1    

        if done:
            if reward > 0.0:
                print("Win")
            else:
                print("Lose")
            break


# # Evaluation
# Take a look at the Q-table. Do the values make sense? Why are some rows still `0`? What is the difference when setting `is_slippery=False`?

# In[12]:


print(Q)


# In[ ]:




