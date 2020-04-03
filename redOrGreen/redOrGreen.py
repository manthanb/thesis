import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt

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

		x = F.softmax(self.layer8(x))

		return x

	def train(self, X, y, λ, ctr, ec):
		optimizer = optim.Adam(self.parameters(), lr=λ)
		optimizer.zero_grad()
		
		out = self.forward(X)

		loss = self.loss_function(out, y)
		if ctr % 5 == 0: print(ec, ctr, loss)
		loss.backward()
		
		optimizer.step()

def getDataForMultiClass(images, labels):
	newLabels = np.zeros((len(labels),3))
	for i in range(len(labels)):
		if labels[i] == 1:
			newLabels[i][0] = 1
		elif labels[i] == 0:
			newLabels[i][1] = 1

	zeroImages = np.ones((500,2,120,120))
	zeroLabels = np.zeros((500,3))
	for i in range(len(zeroLabels)):
		zeroLabels[i][2] = 1

	images = np.concatenate((images, zeroImages))
	labels = np.concatenate((newLabels, zeroLabels))

	return images, labels

def train():

	images = np.load("data_general/images.npy")
	labels = np.load("data_general/labels.npy")

	images, labels = getDataForMultiClass(images, labels)
	images, labels = shuffle(images, labels, random_state=0)

	net = RedOrGreen()

	batchSize = 64
	epochs = 50
	learning_rate = 0.0003
	ctr = 0

	for i in range(epochs):
	
		sampleIdx = set([])
		for j in range(len(images)):
			if j % batchSize == 0: sampleIdx.add(j)

		while len(sampleIdx) > 0:

			start = random.sample(sampleIdx, 1)[0]
			sampleIdx.remove(start)

			end = start + batchSize if start + batchSize <= len(images) else len(images)

			X = images[start:end]
			y = labels[start:end]

			net.train(torch.from_numpy(X).float(), torch.from_numpy(y).float(), learning_rate, ctr, i+1)

			ctr += 1

		# if i % 5 == 0: learning_rate /= 1.5
		torch.save(net.state_dict(), "redOrGreen_multiclass_1.pt")

def test():

	images = np.load("data/images.npy")
	labels = np.load("data/labels.npy")	

	testSamples = [x for x in range(3)]
	testSamples.extend([x for x in range(800,803)])

	net = RedOrGreen()
	net.load_state_dict(torch.load("learned_params/redOrGreen_multiclass_1.pt"))

	for idx in testSamples:
		X = images[idx]
		# X = np.ones((2,120,120))
		X = torch.from_numpy(X).float().unsqueeze(0)
		out = net.forward(X)

		print(idx, out.detach().numpy())

test()
