#!/usr/bin/env python3
import gym
import numpy as np

def __main__():    
    env = gym.make('Ant-v2')

    state = env.reset()
    statespace_shape = env.observation_space.shape[0]
    actionspace_shape = env.action_space.shape[0]

    print("|S|: {}".format(statespace_shape))
    print("|A|: {}".format(actionspace_shape))

    episode_length = 500
    show_first = True

    for i in range(episode_length):
        env.render()
        action = np.random.randn(actionspace_shape,1)
        if show_first:
            print("Action unedited: {}".format(action))

        action = action.reshape((1,-1)).astype(np.float32)
        if show_first:
            print("Action reshaped: {}".format(action))
        action_input = np.squeeze(action,axis=0)
        if show_first:
            print("Action moved again???? {}".format(action_input))
            show_first = False
        state, reward, done, info = env.step(action_input)

if __name__ == '__main__':
    __main__()
