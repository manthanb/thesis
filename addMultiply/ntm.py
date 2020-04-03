import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.autograd import Variable


class NTM(nn.Module):

	def __init__(self, N, W):
	
		super(NTM,self).__init__()

		self.N = N
		self.W = W
		self.Memory = torch.FloatTensor(np.zeros((self.N, self.W))+1e-6)

		self.layer1 = nn.Linear(14,48)
		self.layer2 = nn.Linear(48,72)
		self.layer_ξ = nn.Linear(72,((self.W+1+1+3+1)*2)+(self.W*2))
		self.read_fc = nn.Linear(W, 325)
		self.layer_read = nn.Linear(self.W, 20)
		self.layer_ζ = nn.Linear(72, 3)

		self.read_head = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.W)))

		self.loss_function = nn.BCELoss()

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

	def _separate_alu_params(self, alu_params):
		ρ = F.sigmoid(alu_params[0,0])
		alu_head = F.softmax(alu_params[:,1:])
		return(ρ, alu_head)

	def _interpolate_add_vector(self, ρ, add_vector, v):
		res = ρ*add_vector + (1-ρ)*v
		return res

	def _read_from_memory(self, read_weights):
		return torch.mm(read_weights, self.Memory)

	def _write_to_memory(self, write_weights, erase_vector, add_vector):
		memory_tild = self.Memory * ( torch.tensor(np.zeros((self.N,self.W))+1).float() - torch.mm(torch.transpose(write_weights, 0, 1), erase_vector)) 
		self.Memory = memory_tild + torch.mm(torch.transpose(write_weights, 0, 1), add_vector)

	def forward(self, X, read_weights, write_weights):

		X = F.relu(self.layer1(X))
		X = F.relu(self.layer2(X))
		ξ = self.layer_ξ(X)

		(read_head_params, write_head_params, erase_vector, add_vector) = self._separate_params(ξ)

		(k,g,s,γ,β) = self._get_head_params(read_head_params)
		read_weights = self._address(k, β, g, s, γ, read_weights)
		(k,g,s,γ,β) = self._get_head_params(write_head_params)
		write_weights = self._address(k, β, g, s, γ, write_weights)

		self._write_to_memory(write_weights, erase_vector, add_vector)
		self.read_head = self._read_from_memory(read_weights)
		
		y = self.read_fc(self.read_head)
		out = F.softmax(y,dim=1)

		return (out, read_weights, write_weights)

	def recurrent_forward(self, X):

		(read_weights, write_weights) = self._initialise()

		zeros = torch.tensor(np.zeros((X.size()[0],14))).float()
		out = 0

		for i in range(X.size()[2]):
			(out, read_weights, write_weights) = self.forward(X[:,:,i], read_weights, write_weights)

		for i in range(X.size()[2]):
			(out, read_weights, write_weights) = self.forward(zeros, read_weights, write_weights)

		return out

	def train(self, X, y, mini_batch_size, maxEpoch, learning_rate):

		losses = []
		initial_learning_rate = learning_rate
		epsilon = 0.001
		ctr = 0

		for epoch in range(maxEpoch):

			for i in range(0, len(X)-mini_batch_size, mini_batch_size):

				out = self.recurrent_forward(X[i:i+mini_batch_size])
				loss = self.loss_function(out, y[i:i+mini_batch_size])
				loss.backward(retain_graph=True)

				optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
				optimizer.step()
				optimizer.zero_grad()

				if (i%50 == 0): print(i, epoch, loss)

				ctr += 1
				losses.append(loss.detach().numpy())
				learning_rate = initial_learning_rate / (1 + ctr*epsilon)


			torch.save(self.state_dict(), "ntm.pt")
			np.save("losses/ntm", np.array(losses))

			print(losses)















