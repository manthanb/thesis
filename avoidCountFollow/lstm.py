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
        
		series_output = torch.tensor([])
		for i in range(x.size()[2]):
			h,c,p = self.forward(x[:,:,i],h,c)
			series_output = torch.cat([series_output, p], 0)
        
		return series_output

	def train(self, X, y, maxEpoch, learning_rate, mini_batch_size):

		losses = []
		initial_learning_rate = learning_rate
		epsilon = 0.001
		ctr = 0

		for epoch in range(maxEpoch):

			for i in range(0, len(X)-mini_batch_size, mini_batch_size):

				try:

					s = torch.from_numpy(np.array(X[i:i+mini_batch_size][0])).float().unsqueeze(0)
					l = torch.from_numpy(np.array(y[i:i+mini_batch_size][0])).float()

					s = s.view(s.size()[0], s.size()[2], -1)

					out = self.recurrent_forward(s)

					loss = self.loss_function(out, l)
					loss.backward(retain_graph=True)

					optimizer = torch.optim.Adam( self.parameters(), lr=learning_rate)
					optimizer.step()
					optimizer.zero_grad()

				except Exception as ex:
					print(ex)
					print("exception", i)
					break

				if (i%5 == 0): print(i, epoch, loss)	
				losses.append(loss.detach().numpy())

			torch.save(self.state_dict(), 'params/lstm.pt')
			np.save("data/losses_lstm", np.array(losses))