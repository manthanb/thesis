import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import time

class RedOrGreen(nn.Module):
	
	def __init__(self):

		super(RedOrGreen,self).__init__()

		self.layer1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(8,8), stride=4)
		self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=3, padding=1)
		self.layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1)
		self.layer4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(8,8), stride=1)
		self.layer5 = nn.Linear(512,128)
		self.layer6 = nn.Linear(128, 32)
		self.layer7 = nn.Linear(32, 16)
		self.layer8 = nn.Linear(16, 3)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))

		x = x.view(x.size(0), -1)

		x = F.tanh(self.layer5(x))
		x = F.tanh(self.layer6(x))
		x = F.tanh(self.layer7(x))

		return x

class NTM(nn.Module):

	def __init__(self, N, W, featureExtractorParams):
	
		super(NTM,self).__init__()

		self.N = N
		self.W = W
		self.Memory = torch.FloatTensor(np.zeros((self.N, self.W))+1e-6)

		self.featureExtractor = RedOrGreen()
		self.featureExtractor.load_state_dict(featureExtractorParams)
		for param in self.featureExtractor.parameters():
			self.featureExtractor.requires_grad = False

		self.layer_1 = nn.Linear(18,32)
		self.layer_2 = nn.Linear(32,64)
		self.layer_3 = nn.Linear(64,128)
		self.layer_ξ = nn.Linear(128,((self.W+1+1+3+1)*2)+(self.W*2))
		self.read_fc = nn.Linear(W, 18)

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

	def _read_from_memory(self, read_weights):
		return torch.mm(read_weights, self.Memory)

	def _write_to_memory(self, write_weights, erase_vector, add_vector):
		memory_tild = self.Memory * ( torch.tensor(np.zeros((self.N,self.W))+1).float() - torch.mm(torch.transpose(write_weights, 0, 1), erase_vector)) 
		self.Memory = memory_tild + torch.mm(torch.transpose(write_weights, 0, 1), add_vector)


	def forward(self, X, robotGps, read_weights, write_weights):

		X = self.featureExtractor.forward(X)
		X = torch.cat((X,robotGps),1)

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

		y = self.read_fc(self.read_head)
		out = F.softmax(y,dim=1)
		
		return (out, read_weights, write_weights)

	def recurrent_forward(self, X, y, robotGpsSequence):

		(read_weights, write_weights) = self._initialise()

		series_output = torch.tensor([])
		ones = torch.from_numpy(np.ones((1, 2,120,120))).float()

		for i in range(X.size()[0]):
			ip = X[i,:,:]
			if np.argmax(y[i,:].numpy()) != 0: ip = ones

			(out, read_weights, write_weights) = self.forward(ip, robotGpsSequence[i,:,:], read_weights, write_weights)
			series_output = torch.cat([series_output, out], 0)

		return series_output

	def train(self, X, y, robotGpsSequence, learning_rate):

		out = self.recurrent_forward(X, y, robotGpsSequence)
		loss = self.loss_function(out, y)
		loss.backward(retain_graph=True)

		optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
		optimizer.step()
		optimizer.zero_grad()

		return loss

	def test(self, X, y, imageSequence, robotGpsSequence, learning_rate):

		(read_weights, write_weights) = self._initialise()

		mem = self.Memory.clone()
		plt.matshow(mem.detach().numpy())
		plt.show()

		series_output = torch.tensor([])

		for i in range(X.size()[0]):
			(out, read_weights, write_weights) = self.forward(X[i,:,:], imageSequence[i,:,:,:,:], robotGpsSequence[i,:,:], read_weights, write_weights)
			series_output = torch.cat([series_output, out], 0)

			print(out, y[i])

			if i % 1 == 0 or i == 1:
				mem = self.Memory.clone()
				plt.matshow(mem.detach().numpy())
				plt.show()

		return series_output



