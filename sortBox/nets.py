import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Pick(nn.Module):
	
	def __init__(self):

		super(Pick,self).__init__()

		self.layer1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(8,8), stride=4)
		self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=3, padding=1)
		self.layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1)
		self.layer4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(8,8), stride=1)
		self.layer5 = nn.Linear(512,128)
		self.layer6 = nn.Linear(128, 32)
		self.layer7 = nn.Linear(32, 6)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))

		x = x.view(x.size(0), -1)

		x = F.relu(self.layer5(x))
		x = F.relu(self.layer6(x))
		x = F.softmax(self.layer7(x))

		return x

class RedBin(nn.Module):
	
	def __init__(self):

		super(RedBin,self).__init__()

		self.layer1 = nn.Linear(2, 16)
		self.layer2 = nn.Linear(16,32)
		self.layer3 = nn.Linear(32,64)
		self.layer4 = nn.Linear(64,6)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.tanh(self.layer1(x))
		x = F.tanh(self.layer2(x))
		x = F.tanh(self.layer3(x))
		x = F.softmax(self.layer4(x))

		return x

class GreenBin(nn.Module):
	
	def __init__(self):

		super(GreenBin,self).__init__()

		self.layer1 = nn.Linear(2, 16)
		self.layer2 = nn.Linear(16,32)
		self.layer3 = nn.Linear(32,64)
		self.layer4 = nn.Linear(64,6)

		self.loss_function = nn.BCELoss()

	def forward(self, x):
		x = F.tanh(self.layer1(x))
		x = F.tanh(self.layer2(x))
		x = F.tanh(self.layer3(x))
		x = F.softmax(self.layer4(x))

		return x