import os
import gym
import numpy as np
import random
EPS = 0.1
NUM_ITER = 100000
GAMMA = 0.6
ALPHA = 0.1
all_epochs = []
all_penalties = []


def main():

    env = gym.make("Taxi-v2").env
    if not os.path.isfile('ready.pkl.npy'):
        ##############################################
        #               TRAIN
        ##############################################
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
        reward_mean = []
        for i in range(NUM_ITER):
            state = env.reset()
            done = False
            reward_epoch = []
            while not done:
                a = np.argmax(q_table[state, :]) if random.uniform(0,1) >= EPS else env.action_space.sample()#e-greedy policy
                next_state, reward, done, info = env.step(a)
                next_QValue = np.max(q_table[next_state, :])
                q_table[state, a] = (1-ALPHA) * q_table[state, a] + ALPHA * (reward + GAMMA * next_QValue)
                reward_epoch.append(reward)
                state = next_state
            reward_mean.append(np.mean(reward_epoch))
            if i % 1000 == 0:
                print("episode {}/{}: \t reward {}".format(i, NUM_ITER, np.mean(reward_mean)))
                reward_mean = []
        print("Training finished.\n")
        np.save('ready.pkl.npy', q_table)
    else:
        q_table = np.load('ready.pkl.npy')
    ##############################################
    #               EVALUATE
    ##############################################
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False
        while not done:
            env.render()
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            if reward == -10:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")




if __name__ == '__main__':
    main()