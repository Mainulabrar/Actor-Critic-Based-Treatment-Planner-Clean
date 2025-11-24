import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .myconfig import *

device = iscuda()
if device != "cpu":
    cuda = True

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    # self.state_size = observation_space.shape[0]
    self.state_size = 300
    self.action_size = action_space.n


    self.fc1 = nn.Linear(self.state_size, hidden_size) # (300,64)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size) # (32,32)
    self.fc_actor = nn.Linear(hidden_size, self.action_size) # (32, 4)
    self.fc_critic = nn.Linear(hidden_size, self.action_size) # (32,4)

  def forward(self, x, h):

    # Here, x = state
    x = F.relu(self.fc1(x))
    # print("before h",h)
    # print("before h:x", x)
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    # print("after h", h)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h


# class ActorCriticNetwork(keras.Model):
# 	def __init__(self,observation_shape,num_actions,INPUT_SIZE,episode,name='actor_critic',chkpt_dir='/data/data/Results/GPU2AC/sessionweightless'):
# 		# Structure of both the Actor and Critic Network
# 		super(ActorCriticNetwork,self).__init__()
# 		self._hidden_layer1 = Dense(units=512, activation='relu')
# 		self._hidden_layer2 = Dense(units=256,activation='relu')
# 		self._hidden_layer3 = Dense(units=128,activation='relu')
# 		self._hidden_layer4 = Dense(units= 64,activation='relu')
# 		self.model_name = name
# 		self.checkpoint_dir = chkpt_dir
# 		self.checkpoint_file = os.path.join(self.checkpoint_dir,name+str(episode+1)+'_ac')
#
# 		self.critic = Dense(units=1,activation=None)
# 		# #This is the output layer as it has the activation as 'softmax'
# 		self.actor = Dense(units=num_actions,activation='softmax')
#
#
#
# 	def call(self,state):
# 		value = self._hidden_layer1(state)
# 		value = self._hidden_layer2(value)
# 		value = self._hidden_layer3(value)
# 		value = self._hidden_layer4(value)
#
# 		critic = self.critic(value)
# 		actor = self.actor(value)
#
# 		return critic,actor




# class ActorCriticNetwork(nn.Module):
#     def __init__(self,INPUT_SIZE,num_actions = actionnum()):
#         super(ActorCriticNetwork, self).__init__()
#         self.affine = nn.Linear(INPUT_SIZE, 64)
#         # self.affine2 = nn.Linear(64, 32)
#
#         self.action_layer = nn.Linear(64,num_actions)
#         self.value_layer = nn.Linear(64, 1)
#
#         self.logprobs = []
#         self.state_values = []
#         self.rewards = []
#
#         self.device = iscuda()
#
#         self.to(self.device)
#
#     def forward(self, state, boolean=False):
#         #CUDA: Comment below line
#         # state = torch.from_numpy(state).float()
#         state = F.relu(self.affine(state))
#         # print("state = F.relu(self.affine(state))",state.device)
#         # state = F.relu(self.affine2(state1))
#         state_value = self.value_layer(state)
#         # print("state_value = self.value_layer(state)", state_value.device)
#         action_probs = F.softmax(self.action_layer(state))
#         # print("action_probs = F.softmax(self.action_layer(state))", action_probs.device)
#
#         # print("action_probs",action_probs)
#         # print("preferred action:", np.argmax(action_probs.detach().numpy()))
#         action_distribution = Categorical(action_probs)
#
#
#         if boolean == True:
#             if cuda:
#                 action = torch.argmax(action_probs)
#             else:
#                 action1 = np.argmax(action_probs.detach().numpy())
#                 # This is trying to change the action to numpy array
#                 new_array = np.array([action1], dtype=np.int32)
#                 # This then changes to tensor array from numpy array
#                 action = torch.from_numpy(new_array)
#             #             print("action",action)
#
#         else:
#             action = action_distribution.sample()
#             # print("chosen action:",action)
#
#         self.logprobs.append(action_distribution.log_prob(action))
#         # print("action_distribution.log_prob(action)",action_distribution.log_prob(action).device)
#         self.state_values.append(state_value)
#         # print("state_value",state_value.device)
#         return action.item()
#
#     def calculateLoss(self, discount, check):
#         device = iscuda()
#         # print("device = iscuda()",device)
#
#         # calculating discounted rewards:
#         rewards = []
#         dis_reward = 0
#         # print("len(self.rewards)",len(self.rewards))
#         # rewards = torch.empty(0,len(self.rewards))
#         # dis_reward = torch.tensor([0])
#         # dis_reward = dis_reward.to(device)
#
#
#         # if True:
#         # print("self.rewards_before discounting", self.rewards)
#
#         # print("self.rewards[::-1]", self.rewards[::-1])
#         for reward in self.rewards[::-1]:
#             # print("reward",reward)
#             # print("gamma",discount)
#             # print("dis_reward",dis_reward)
#             # print("gamma * dis_reward",discount * dis_reward)
#             dis_reward = reward + discount * dis_reward
#             # print("dis_reward = reward + discount * dis_reward",dis_reward)
#             # product = torch.mul(gamma,dis_reward)
#             # dis_reward = torch.add(reward,product)
#             rewards.insert(0,dis_reward)
#             # print("rewards.insert(0,dis_reward)",rewards)
#         # print("self.rewards_after_discounting:", rewards)
#
#
#         # normalizing the rewards:
#         # rewards = np.array(rewards) #This was kept because of warning: Creating a tensor from a list of numpy.ndarrays is slow.
#         # print("rewards = np.array(rewards)",rewards)
#         rewards = torch.tensor(rewards)#############################################################
#         # print("rewards = torch.tensor(rewards)",rewards)
#         # rewards.to(device)
#         # print("rewards", rewards)
#         num = torch.numel(rewards)
#
#         # num = 0
#
#         # print("num = torch.numel(rewards)",num)
#         # print("rewards before normalizing", rewards)
#         if num == 1:
#           rewards = torch.tensor([0.0])
#         else:
#             rewards = (rewards - rewards.mean()) / (rewards.std())
#           # print("rewards.mean()",rewards.mean())
#           # print("rewards.std()",rewards.std())
#           # print("rewards - rewards.mean()",rewards - rewards.mean())
#           # print("(rewards - rewards.mean()) / (rewards.std())",(rewards - rewards.mean()) / (rewards.std()))
#
#
#           # rewards = (rewards - rewards.mean()) / (rewards.std())
#
#
#         if cuda:
#             rewards = rewards.to(device)
#         # print("rewards", rewards)
#         # print("rewards type", type(rewards))
#         loss = 0
#         # print("outside the loop")
#
#
#         for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
#             # lobprob = logprob.to("cpu")
#             # print("hello")
#             # print("reward",reward.device)
#             advantage = reward - value.item()
#
#             # print("advantage",advantage.device)
#             # print("logprob",logprob.device)
#             # print("reward", reward.device)
#             action_loss = -logprob * advantage
#             # print("value",value.shape)
#             value_loss = F.smooth_l1_loss(value, reward)
#             # print("value_loss before",value_loss.shape)
#             new_shape = (1,1)
#             value_loss = value_loss.view(new_shape)
#             # print("value_loss after", value_loss.shape)
#             loss += (action_loss + value_loss)
#             # if num == 1:
#             #   print("advantage",advantage)
#             #   print("action_loss",action_loss)
#             #   print("value_loss",value_loss)
#             #   print("loss",loss)
#         return loss
#
#     def clearMemory(self):
#         del self.logprobs[:]
#         del self.state_values[:]
#         del self.rewards[:]
