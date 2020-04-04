import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.autograd import Variable

from nets import Obstacle, WallCW, WallACW

class HNM(nn.Module):

	def __init__(self, N, W, networks, params):
	
		super(HNM,self).__init__()

		self.N = N
		self.W = W
		self.Memory = torch.FloatTensor(np.zeros((self.N, self.W))+1e-6)

		self.layer_1 = nn.Linear(14,32)
		self.layer_2 = nn.Linear(32,64)
		self.layer_3 = nn.Linear(64,128)
		self.layer_ξ = nn.Linear(128,((self.W+1+1+3+1)*2)+(self.W*2))
		self.layer_ζ_1 = nn.Linear(28, 16)
		self.layer_ζ_2 = nn.Linear(16, 3)

		self.read_head = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.W)))

		self.loss_function = nn.BCELoss()

		self.networks = networks
		self.params = params
		for i in range(len(self.networks)):
			self.networks[i].load_state_dict(self.params[i])

	def _initialise(self):
		self.Memory = torch.FloatTensor(np.zeros((self.N, self.W))+1e-6)
		self.read_head = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.W)))
		read_weights = write_weights = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.N)))
		return (read_weights, write_weights)

	def _separate_params(self, ξ):
		read_head_params = ξ[0, :self.W+6].view(1,-1)
		write_head_params = ξ[0, self.W+6:(self.W+6)*2].view(1,-1)
		erase_vector = F.sigmoid(ξ[0, (self.W+6)*2:(self.W+6)*2+self.W].view(1,-1))
		add_vector = F.tanh(ξ[0, (self.W+6)*2+self.W:(self.W+6)*2+(self.W)*2].view(1,-1))

		return (read_head_params, write_head_params, erase_vector, add_vector)

	def _get_head_params(self, head_params):
		k = F.tanh(head_params[0,:self.W].view(1,-1))
		g = F.sigmoid(head_params[0, self.W])
		s = F.softmax(head_params[0, self.W+1:self.W+1+3])
		γ = 1 + F.softplus(head_params[0, self.W+1+3])
		β = F.softplus(head_params[0, self.W+1+3+1])

		return (k,g,s,γ,β)

	def _address_by_content(self, k, β):
		sim = F.cosine_similarity(self.Memory+1e-16, k+1e-16, dim=-1)
		sim = sim * β
		w_c = F.softmax(sim)
		return w_c

	def _interpolate(self, w_c, g, w_prev):
		w_g = g*w_c + (1-g)*w_prev
		return w_g

	def _random_shift(self, w_g, s):
		w_modulo_unrolled = torch.cat([w_g[:, -1:], w_g, w_g[:, :1]], 1)
		s = s.view(1, 1,-1)
		w_modulo_unrolled = w_modulo_unrolled.view(1,1,-1)
		return F.conv1d( w_modulo_unrolled, s)[0,:,:]

	def _sharpen(self, w_tild, γ):
		w = w_tild ** γ
		w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
		return w

	def _address(self, k, β, g, s, γ, w_prev):
		w_c = self._address_by_content(k, β)
		w_g = self._interpolate(w_c, g, w_prev)
		w_r = self._random_shift(w_g, s)
		w = self._sharpen(w_r, γ)
		
		return w

	def _read_from_memory(self, read_weights):
		return torch.mm(read_weights, self.Memory)

	def _write_to_memory(self, write_weights, erase_vector, add_vector):
		memory_tild = self.Memory * ( torch.tensor(np.zeros((self.N,self.W))+1).float() - torch.mm(torch.transpose(write_weights, 0, 1), erase_vector)) 
		self.Memory = memory_tild + torch.mm(torch.transpose(write_weights, 0, 1), add_vector)

	def _alu_compute(self, X, P):
		X = X[:,:8]
		out = P[0][0]*self.networks[0].forward(X) + P[0][1]*self.networks[1].forward(X) + P[0][2]*self.networks[2].forward(X)
		return out

	def forward(self, X, read_weights, write_weights):

		X0 = X.clone()
		X = F.tanh(self.layer_1(X))
		X = F.tanh(self.layer_2(X))
		X = F.tanh(self.layer_3(X))
		ξ = self.layer_ξ(X)

		(read_head_params, write_head_params, erase_vector, add_vector) = self._separate_params(ξ)

		(k,g,s,γ,β) = self._get_head_params(read_head_params)
		read_weights = self._address(k, β, g, s, γ, read_weights)
		(k,g,s,γ,β) = self._get_head_params(write_head_params)
		write_weights = self._address(k, β, g, s, γ, write_weights)

		self._write_to_memory(write_weights, erase_vector, add_vector)
		self.read_head = self._read_from_memory(read_weights)
		
		alu_input = torch.cat([X0, self.read_head], 1)
		P = F.tanh(self.layer_ζ_1(alu_input))
		P = F.softmax(self.layer_ζ_2(P))

		out = self._alu_compute(X0, P)
		
		return (out, read_weights, write_weights)

	def recurrent_forward(self, X):

		(read_weights, write_weights) = self._initialise()

		series_output = torch.tensor([])

		for i in range(X.size()[1]):
			(out, read_weights, write_weights) = self.forward(X[:,i,:], read_weights, write_weights)
			series_output = torch.cat([series_output, out], 0)

		return series_output

	def train(self, X, y, mini_batch_size):

		learning_rate = 0.0003
		losses = []
		# 4
		for epoch in range(20):

			for i in range(0, len(X)-mini_batch_size, mini_batch_size):

				try:

					s = torch.from_numpy(np.array(X[i:i+mini_batch_size][0])).float().unsqueeze(0)
					l = torch.from_numpy(np.array(y[i:i+mini_batch_size][0])).float()

					out = self.recurrent_forward(s)
					loss = self.loss_function(out, l)
					loss.backward(retain_graph=True)

					optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
					optimizer.step()
					optimizer.zero_grad()

				except Exception as ex:
					print(ex)
					print("exception", i)
					continue

				if (i%10 == 0): print(i, epoch, loss)
				losses.append(loss.detach().numpy())

			torch.save(self.state_dict(), 'hnm.pt')

			np.save("data/losses_hnm", np.array(losses))
			if epoch % 2 == 0 and epoch >0 and learning_rate > 0.0001: learning_rate /= 2
