#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pomdp as pd
from math import exp
from statistics import mean
from scipy.stats import norm
from copy import deepcopy

class Actor(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=1e-5, debug=False):
        super(Actor, self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size * 2 )
        self.actor_layer2 = nn.Linear(hidden_size * 2, hidden_size) 
        self.actor_means = nn.Linear(hidden_size, actionspace_size)
        self.actor_devs = nn.Linear(hidden_size, actionspace_size)

    def forward(self,state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(action_distribution)

        means = self.actor_means(F.relu(action_distribution))
        log_deviations = self.actor_devs(F.relu(action_distribution))


        if self.debug:
            print("Means: {}".format(means))
            print("Log Deviations: {}".format(log_deviations))
        return means, log_deviations

 
class Critic(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=1e-5, debug=False):
        super(Critic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.critic_layer1 = nn.Linear(statespace_size, hidden_size * 2)
        self.critic_layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_layer3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        state_estimate = F.relu(self.critic_layer1(state))
        state_estimate = F.relu(self.critic_layer2(state_estimate))
        state_estimate = self.critic_layer3(state_estimate)
        if self.debug:
            print("State value estimate: {}".format(state_estimate))
        return state_estimate 



class ActorCritic(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=1e-5, debug=False):
        super(ActorCritic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size * 2 )
        self.actor_layer2 = nn.Linear(hidden_size * 2, hidden_size)

        self.actor_means = nn.Linear(hidden_size, actionspace_size)
        self.actor_devs = nn.Linear(hidden_size, actionspace_size)
        self.critic_layer1 = nn.Linear(statespace_size, hidden_size * 2)
        self.critic_layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_layer3 = nn.Linear(hidden_size, 1)


    def action_distribution(self, state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(action_distribution)

        means = self.actor_means(F.relu(action_distribution))
        log_deviations = self.actor_devs(F.relu(action_distribution))


        if self.debug:
            print("Means: {}".format(means))
            print("Log Deviations: {}".format(log_deviations))
        return means, log_deviations

    def state_estimate(self, state):
        state_estimate = F.relu(self.critic_layer1(state))
        state_estimate = F.relu(self.critic_layer2(state_estimate))
        state_estiamte = self.critic_layer3(state_estimate)
        if self.debug:
            print("State value estimate: {}".format(state_estimate))
        return state_estimate 

    def forward(self, state):
        if self.debug:
            print("Input state: {}".format(state))

        state_estimate = self.state_estimate(state)
        action_distribution = self.action_distribution(state)

        return action_distribution, state_estimate

class PPO:
    def __init__(self,  statespace_size, actionspace_size, hidden_size, device, discount, learning_rate=1e-5, debug=False, lambda_return = 0.95):

        self.device = device
        self.actor= Actor(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5).to(device)
        self.critic = Critic(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5).to(device)
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.discount_rate = discount

        self.lambda_return = lambda_return
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.previous_policy = None 
        self.previous_optimizer = None
        self.learning_rate = learning_rate
        self.average_reward = 0


    def actor_error(self, reward, state_value, next_state_value):
        error = (reward + (self.discount_rate*next_state_value) ) - state_value
        return error

    def critic_error(self, reward, state_value, timestep):
        self.average_reward = self.average_reward + (self.average_reward - reward)/timestep
        #print("{} - {}".format(state_value, self.average_reward))
        return state_value - self.average_reward


def __main__():    
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256
    discount = 0.99
    clipping_epsilon = 0.2
    env = gym.make('Hopper-v2')
    # Let's start simple and do the MDP before we get the rest of this 
    pomdp = pd.PartiallyObservableEnv(env, seed=42)
    num_iterations = 1000

    state = torch.tensor(pomdp.reset()).float().to(device)
    statespace_size = pomdp.env.observation_space.shape[0]
    actionspace_size = pomdp.env.action_space.shape[0]

    print("|S|: {}".format(statespace_size))
    print("|A|: {}".format(actionspace_size))

    horizon = 500
    show_first = True

    agent = PPO(statespace_size, actionspace_size, hidden_size, device, discount)
    agent.actor.train()
    agent.critic.train()
    
    for iteration in range(num_iterations):
        distributions = []
        length = 0
        error_list = []
        ratio_list = []
        reward_list = []        

        previous_reward = None
        previous_value = None
        error = None

        for step in range(horizon):
            action_distribution = agent.actor.forward(state)
            value = agent.critic.forward(state)

            if previous_reward is not None:
                print(iteration)
                actor_error = agent.actor_error(previous_reward, previous_value, value)
                critic_error = agent.critic_error(previous_reward, previous_value, step)
                error_list.append([actor_error, critic_error])

            means = action_distribution[0].detach().cpu().numpy()
            log_deviations =  action_distribution[1]
            deviations = [exp(dev)**2 for dev in log_deviations]
            distributions.append([means, log_deviations])
            action = np.random.normal(means, deviations , actionspace_size)

            next_state, reward, done, info = pomdp.step(action)
            reward_list.append(reward)

            ratio = None
            equal = True
            if agent.previous_policy is not None:
                for p1, p2 in zip(agent.previous_policy.parameters(), agent.actor.parameters()):
                    if p1.data.ne(p2.data).sum() > 0:
                       equal = False
                if not equal:
                    print("why tho")
                
                past_distribution = agent.previous_policy.forward(state) 
                past_deviations = past_distribution[1].detach()
                ratio = log_deviations - past_deviations
                ratio = torch.exp(ratio)
            else:
                ratio = torch.ones(len(log_deviations)).to(device)
                
            
            clipped_ratio = torch.clamp(ratio,(1 - clipping_epsilon),(1 + clipping_epsilon))
            ratio = torch.tensor([min(ratio[i],clipped_ratio[i]) for i in range(len(ratio))],requires_grad=True)

            if not torch.all( torch.eq(ratio,torch.ones(len(log_deviations))) ):
                print("gasp")
            previous_reward = reward
            previous_value = value

            state = torch.tensor(next_state).float().to(device)
            ratio_list.append(ratio)

        agent.actor.eval()
        torch.save({
                    'actor' : agent.actor.state_dict(), 
                    'optimizer' : agent.actor_optimizer.state_dict()
                    }, ".tmp")

        agent.previous_policy = Actor(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5).to(device)
        agent.previous_optimizer = optim.Adam(agent.previous_policy.parameters(), lr = agent.learning_rate)
        previous = torch.load(".tmp")
        agent.previous_policy.load_state_dict(previous['actor'])
        agent.previous_optimizer.load_state_dict(previous['optimizer'])
        agent.previous_policy.eval()


        reversed_advantages = []
        critic_loss = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)
        advantage = 0

        for backwards_step in reversed(error_list):
            advantage = advantage * agent.lambda_return * agent.discount_rate 
            advantage += backwards_step[0]
            critic_loss += torch.mul(backwards_step[1], backwards_step[1])
            reversed_advantages.append(advantage)
                            
        critic_loss = critic_loss / len(error_list)

        loss_list = []
        actor_loss = None

        for i in range(len(reversed_advantages)):
           advantage = reversed_advantages[len(reversed_advantages) - i - 1]
           ratio = ratio_list[i]
           loss_list.append(torch.tensor([ advantage * val for val in ratio], requires_grad=True).to(device))


        loss_list = torch.tensor([torch.sum(lst) for lst in loss_list], requires_grad=True).to(device)
        
        actor_loss = sum(loss_list)
        actor_loss = actor_loss/len(loss_list)
        neg_actor_loss = -1 * actor_loss
        agent.actor_optimizer.zero_grad()
        neg_actor_loss.backward(retain_graph=True)
        agent.actor_optimizer.step()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        #print(agent.average_reward)
        if iteration % 50 == 0:
            print("{}: {}".format(iteration,agent.average_reward))
        agent.average_reward = 0
            

if __name__ == '__main__':
    __main__()
