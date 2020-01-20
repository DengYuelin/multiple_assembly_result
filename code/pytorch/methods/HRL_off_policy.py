import numpy as np
import torch
import torch.nn as nn
import random
import glob
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
from ..utils.model import Actor1D, ActorList, Critic, Option, OptionValue

if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LongTensor = torch.cuda.LongTensor


class HRLACOP(object):
	def __init__(self, state_dim, action_dim, max_action, option_num=2,
				 entropy_coeff=0.1, c_reg=1.0, c_ent=4, option_buffer_size=5000,
				 action_noise=0.2, policy_noise=0.2, noise_clip = 0.5, use_option_net=True):

		self.actor = ActorList(state_dim, action_dim, max_action, option_num).to(device)
		self.actor_target = ActorList(state_dim, action_dim, max_action, option_num).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		if use_option_net:
			self.option = OptionValue(state_dim, option_num).to(device)
			self.option_target = OptionValue(state_dim, option_num).to(device)
			self.option_target.load_state_dict(self.option.state_dict())
			self.option_optimizer = torch.optim.Adam(self.option.parameters())

		self.use_option_net = use_option_net
		self.max_action = max_action
		self.it = 0

		self.entropy_coeff = entropy_coeff
		self.c_reg = c_reg
		self.c_ent = c_ent

		self.option_buffer_size = option_buffer_size
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.option_num = option_num
		self.k = 40
		self.action_noise = action_noise
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.q_predict = np.zeros(self.option_num)
		self.option_val = 0

	def train(self,
			  replay_buffer_lower,
			  replay_buffer_higher,
			  batch_size_lower=100,
			  batch_size_higher=100,
              discount_higher=0.99,
              discount_lower=0.99,
              tau=0.005,
              policy_freq=2):
		self.it += 1
		state, action, option, target_q, current_q_value = \
			self.calc_target_q(replay_buffer_lower, batch_size_lower, discount_lower, is_on_poliy=False)

		# ================ Train the critic ============================================= #
		self.train_critic(state, action, target_q)

		# Delayed policy updates
		if self.it % policy_freq == 0:
			# ================ Train the actor =============================================#
			# off-policy learning :: sample state and option pair from replay buffer
			self.train_actor(state, current_q_value)
			# ===============================================================================#

			# update the frozen target networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		# Delayed option updates
		if self.use_option_net:
			if self.it % self.option_buffer_size == 0:
				for _ in range(self.option_buffer_size):
					state, option, target_q = \
						self.calc_target_option_q(replay_buffer_higher, batch_size_higher, discount_higher, is_on_poliy=True)
					self.train_option(state, action, target_q)

	def train_critic(self, state, action, target_q):
		'''
		Calculate the loss of the critic and train the critic :: TD3
		'''
		current_q1, current_q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_q1, target_q) + \
					  F.mse_loss(current_q2, target_q)

		# Three steps of training net using PyTorch:
		self.critic_optimizer.zero_grad()  # 1. Clear cumulative gradient
		critic_loss.backward()  # 2. Back propagation
		self.critic_optimizer.step()  # 3. Update the parameters of the net

	def train_actor(self, current_q_value):
		'''
		Calculate the loss of the actor and train the actor
		'''
		actor_loss = - current_q_value.mean()

		# Three steps of training net using PyTorch:
		self.actor_optimizer.zero_grad()  # 1. Clear cumulative gradient
		actor_loss.backward()  # 2. Back propagation
		self.actor_optimizer.step()  # 3. Update the parameters of the net

	def train_option(self, state, option, target_q):
		'''
		Calculate the loss of the option and train the option ：：：DQN
		'''

		current_q, _ = self.option(state)
		current_q = current_q.gather(1, option)
		option_loss = F.mse_loss(current_q, target_q)

		self.option_optimizer.zero_grad()  # 1. Clear cumulative gradient
		option_loss.backward()  # 2. Back propagation
		self.option_optimizer.step()  # 3. Update the parameters of the net

	def calc_target_q(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=False):
		'''
		calculate q value for low-level policies
		:param replay_buffer:
		:param batch_size:
		:param discount:
		:param is_on_poliy:
		:return:
		'''
		policy_noise = self.policy_noise
		noise_clip = self.noise_clip
		if is_on_poliy:
			x, y, u, o, o_1, r, d = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, u, o, o_1, r, d = \
				replay_buffer.sample(batch_size)

		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		option = torch.FloatTensor(o).to(device)
		next_state = torch.FloatTensor(y).to(device)

		# need to be revised
		next_option = torch.FloatTensor(o_1).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)

		# inject noise and select next action
		noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
		noise = noise.clamp(-noise_clip, noise_clip)
		next_action = (self.actor_target(next_state)[torch.arange(next_state.shape[0]), :, next_option]
					   + noise).clamp(-self.max_action, self.max_action)

		# for updating q-value
		target_q1, target_q2 = self.critic_target(next_state, next_action)
		target_q = torch.min(target_q1, target_q2)
		target_q = reward + (done * discount * target_q)

		# for updating policy
		action_1 = self.actor(state, option)
		current_q_value = self.critic(state, action_1)
		return state, action, option, target_q, current_q_value

	def calc_target_option_q(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=True):
		'''
		calculate option value
		:param replay_buffer:
		:param batch_size:
		:param discount: option policy discount
		:param is_on_poliy:
		:return:
		'''
		if is_on_poliy:
			x, y, o, r = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, o, r = \
				replay_buffer.sample(batch_size)

		state = torch.FloatTensor(x).to(device)
		next_state = torch.FloatTensor(y).to(device)
		option = torch.FloatTensor(o).to(device)
		reward = torch.FloatTensor(r).to(device)

		option_next = self.select_option(next_state)
		next_q = self.option_target(next_state).gather(1, option_next)
		target_q = reward + discount * next_q
		return state, option, target_q

	def calc_advantage_value(self, state):
		'''
		calculate adv for upper policies
		:param reward:
		:param state:
		:param next_state:
		:param gamma:
		:return:
		'''
		option_value, _ = self.option(state)
		advantage_value = option_value - torch.mean(option_value)

		# calculate the low-level auxiliary reward
		low_level_reward = advantage_value / self.k
		return advantage_value, low_level_reward

	def select_option(self, state):
		'''
		select new option every N or 2N steps
		'''
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		_, option_num = self.option_target(state)
		return option_num.data.max(1)[1].view(1, 1).cpu().data.numpy().flatten()

	def select_action(self, state, option):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action = self.actor(state)[torch.arange(state.shape[0]), :, option]
		return action.cpu().data.numpy().flatten()

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))
