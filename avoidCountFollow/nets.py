import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Obstacle(nn.Module):
	
	def __init__(self):

		super(Obstacle,self).__init__()

		self.layer1 = nn.Linear(8,32)
		self.layer2 = nn.Linear(32,16)
		self.layer3 = nn.Linear(16,3)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.softmax(self.layer3(x))
		return x

	def train(self, X, y, λ, miniBatch, epochs):

		for i in range(epochs):

			for j in range(len(X)-miniBatch):

				optimizer = optim.SGD(self.parameters(), lr=λ)
				optimizer.zero_grad()
				
				out = self.forward(X[j:j+miniBatch])
				
				loss = self.loss_function(out, y[j:j+miniBatch])
				if j % 1000 == 0: print(j,loss)
				loss.backward()	
				
				optimizer.step()

			torch.save(self.state_dict(), 'params/move.pt')
			if i % 2 == 0: λ /= 1.5
			# print("epoch:", i+1, λ)

			print("epoch:", i+1)

class WallCW(nn.Module):
	
	def __init__(self):

		super(WallCW,self).__init__()

		self.layer1 = nn.Linear(8,32)
		self.layer2 = nn.Linear(32,16)
		self.layer3 = nn.Linear(16,3)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.softmax(self.layer3(x))
		return x

	def train(self, X, y, λ, miniBatch, epochs):

		for i in range(epochs):

			for j in range(len(X)-miniBatch):

				optimizer = optim.SGD(self.parameters(), lr=λ)
				optimizer.zero_grad()
				
				out = self.forward(X[j:j+miniBatch])
				
				loss = self.loss_function(out, y[j:j+miniBatch])
				if j % 100 == 0: print(j,loss)
				loss.backward()	
				
				optimizer.step()

			torch.save(self.state_dict(), 'params/cw.pt')

			# if i % 2 == 0: λ /= 1.5
			# print("epoch:", i+1, λ)

			print("epoch:", i+1)

class WallACW(nn.Module):
	
	def __init__(self):

		super(WallACW,self).__init__()

		self.layer1 = nn.Linear(8,32)
		self.layer2 = nn.Linear(32,16)
		self.layer3 = nn.Linear(16,3)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.softmax(self.layer3(x))
		return x

	def train(self, X, y, λ, miniBatch, epochs):

		for i in range(epochs):

			for j in range(len(X)-miniBatch):

				optimizer = optim.SGD(self.parameters(), lr=λ)
				optimizer.zero_grad()
				
				out = self.forward(X[j:j+miniBatch])
				
				loss = self.loss_function(out, y[j:j+miniBatch])
				if j % 100 == 0: print(j,loss)
				loss.backward()	
				
				optimizer.step()

			torch.save(self.state_dict(), 'params/acw.pt')

			# if i % 2 == 0: λ /= 1.5
			# print("epoch:", i+1, λ)

			print("epoch:", i+1)