import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class LSTM(nn.Module):

	def __init__(self, input_dims, hidden_dims, output_dims, batch_size):
		super(LSTM,self).__init__()

		self.input_dims = input_dims
		self.hidden_dims = hidden_dims
		self.output_dims = output_dims
		self.batch_size = batch_size
        
		self.lstm = nn.LSTMCell(input_dims, hidden_dims)
		self.output = nn.Linear(hidden_dims,output_dims)
        
		self.h0 = torch.FloatTensor(np.random.randn(1, hidden_dims))
		self.c0 = torch.FloatTensor(np.random.randn(1, hidden_dims))
		self.loss_function = nn.BCELoss()
        
	def forward(self, x, h, c): 
		h,c = self.lstm(x,(h,c))
		p = F.softmax(self.output(c),dim=1)
        
		return h,c,p
    
	def recurrent_forward(self,x):
		h = self.h0.expand(x.size()[0],self.hidden_dims)
		c = self.c0.expand(x.size()[0],self.hidden_dims)
		zvec = torch.tensor(np.zeros((x.size()[0],x.size()[1]))).float()
        
		for i in range(x.size()[2]):
			h,c,p = self.forward(x[:,:,i],h,c)
        
		for i in range(x.size()[2]):
			h,c,p = self.forward(zvec,h,c)
        
		return p

	def train(self, X, y, maxEpoch, learning_rate):

		losses = []
		initial_learning_rate = learning_rate
		epsilon = 0.001
		ctr = 0

		for epoch in range(maxEpoch):

			for i in range(0, len(X)-self.batch_size, self.batch_size):

				out = self.recurrent_forward(X[i:i+self.batch_size])

				loss = self.loss_function(out, y[i:i+self.batch_size])
				loss.backward(retain_graph=True)

				optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
				optimizer.step()
				optimizer.zero_grad()

				if (i%50 == 0): print(i, epoch, loss)

				ctr += 1
				learning_rate = initial_learning_rate / (1 + ctr*epsilon)
				losses.append(loss.detach().numpy())
				
			torch.save(self.state_dict(), "lstm.pt")
			np.save("losses/lstm", np.array(losses))
