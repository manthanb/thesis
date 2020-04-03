import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ALU(nn.Module):

	def __init__(self):

		super(ALU,self).__init__()

		self.layer1 = nn.Linear(40,110)
		self.layer2 = nn.Linear(110,190)
		self.layer3 = nn.Linear(190,270)
		self.layer4 = nn.Linear(270,325)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.softmax(self.layer4(x))
		return x

class HNM(nn.Module):

	def __init__(self, N, W, add_params, mul_params):
	
		super(HNM,self).__init__()

		self.N = N
		self.W = W
		self.Memory = torch.FloatTensor(np.zeros((self.N, self.W))+1e-6)

		self.layer1 = nn.Linear(14,48)
		self.layer2 = nn.Linear(48,72)
		self.layer_ξ = nn.Linear(72,((self.W+1+1+3+1)*2)+(self.W*2))
		self.layer_v = nn.Linear(325, 20)
		self.layer_ζ = nn.Linear(72, 3)

		self.read_head = F.sigmoid(torch.FloatTensor(np.random.randn(1,self.W)))

		self.loss_function = nn.BCELoss()

		self.add_params = add_params
		self.mul_params = mul_params
		self.alu_add_model = ALU()
		self.alu_mul_model = ALU()
		self.alu_add_model.load_state_dict(self.add_params)
		self.alu_mul_model.load_state_dict(self.mul_params)


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

	def _alu_compute(self, alu_head, alu_input):
		alu_add_out = self.alu_add_model.forward(alu_input)
		alu_mul_out = self.alu_mul_model.forward(alu_input)

		out = alu_head[0,0]*alu_add_out + alu_head[0,1]*alu_mul_out

		return out

	def _read_from_memory(self, read_weights):
		return torch.mm(read_weights, self.Memory)

	def _write_to_memory(self, write_weights, erase_vector, add_vector):
		memory_tild = self.Memory * ( torch.tensor(np.zeros((self.N,self.W))+1).float() - torch.mm(torch.transpose(write_weights, 0, 1), erase_vector)) 
		self.Memory = memory_tild + torch.mm(torch.transpose(write_weights, 0, 1), add_vector)

	def forward(self, X, read_weights, write_weights):

		X = self.layer1(X)
		X = self.layer2(X)

		ξ = self.layer_ξ(X)
		ζ = self.layer_ζ(X)

		(read_head_params, write_head_params, erase_vector, add_vector) = self._separate_params(ξ)

		(k,g,s,γ,β) = self._get_head_params(read_head_params)
		read_weights = self._address(k, β, g, s, γ, read_weights)
		(k,g,s,γ,β) = self._get_head_params(write_head_params)
		write_weights = self._address(k, β, g, s, γ, write_weights)

		new_read_head = self._read_from_memory(read_weights)
		
		alu_input = torch.cat([self.read_head, new_read_head], 1)
		(ρ, alu_head) = self._separate_alu_params(ζ)

		out = self._alu_compute(alu_head, alu_input)
		v = self.layer_v(out)

		add_vector = self._interpolate_add_vector(ρ, add_vector.clone(), v)
		self._write_to_memory(write_weights, erase_vector, add_vector)
		self.read_head = new_read_head

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

	def train(self, X, y, mini_batch_size):

		learning_rate = 0.0009
		losses = []

		for epoch in range(25):

			for i in range(0, len(X)-mini_batch_size, mini_batch_size):

				out = self.recurrent_forward(X[i:i+mini_batch_size])
				loss = self.loss_function(out, y[i:i+mini_batch_size])
				loss.backward(retain_graph=True)

				optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
				optimizer.step()
				optimizer.zero_grad()

				if (i%50 == 0): print(i, epoch, loss)

				losses.append(loss.detach().numpy())

			torch.save(self.state_dict(), 'hnm_arch_1.pt')
			np.save("losses/hnm_arch_1", np.array(losses))

			if epoch % 10 == 0: learning_rate /= 1.5












