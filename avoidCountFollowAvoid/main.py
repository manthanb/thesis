import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable

from nets import Obstacle, WallCW, WallACW
from data_lib import getData
from harvard_machine import HNM
from ntm import NTM
from lstm import LSTM

def trainHNM():
	obstacle, wall_cw, wall_awc = Obstacle(), WallCW(), WallACW()
	obstacle_params, wall_cw_params, wall_acw_params = torch.load('program_memory/move.pt'), torch.load('program_memory/cw.pt'), torch.load('program_memory/acw.pt')

	networks = [obstacle, wall_cw, wall_awc]
	params = [obstacle_params, wall_cw_params, wall_acw_params]

	hnm = HNM(10, 14, networks, params)

	X, y = [], []
	for i in range(10):
		tempX, tempY = getData("data/observations_"+str(i*500)+".npy", "data/actions_"+str(i*500)+".npy")
		X.extend(tempX)
		y.extend(tempY)

	print(len(X), len(y))

	hnm.train(X, y, 1)

def trainNTM():
	ntm = NTM(10, 14)

	X, y = [], []
	for i in range(10):
		tempX, tempY = getData("data/observations_"+str(i*500)+".npy", "data/actions_"+str(i*500)+".npy")
		X.extend(tempX)
		y.extend(tempY)

	print(len(X), len(y))

	ntm.train(X, y, 1)

def trainLSTM():
	lstm = LSTM(14, 64, 3, 1)

	X, y = [], []
	for i in range(10):
		tempX, tempY = getData("data/observations_"+str(i*500)+".npy", "data/actions_"+str(i*500)+".npy")
		X.extend(tempX)
		y.extend(tempY)

	print(len(X), len(y))

	lstm.train(X, y, maxEpoch=10, learning_rate=0.0006, mini_batch_size=1)

def compare():

	obstacle, wall_cw, wall_awc = Obstacle(), WallCW(), WallACW()
	obstacle_params, wall_cw_params, wall_acw_params = torch.load('program_memory/move.pt'), torch.load('program_memory/cw.pt'), torch.load('program_memory/acw.pt')
	networks = [obstacle, wall_cw, wall_awc]
	params = [obstacle_params, wall_cw_params, wall_acw_params]
	hnm = HNM(10, 14, networks, params)
	hnm.load_state_dict(torch.load('learned_params/hnm.pt'))

	ntm = NTM(10, 14)
	ntm.load_state_dict(torch.load('learned_params/ntm.pt'))

	lstm = LSTM(14, 64, 3, 1)
	lstm.load_state_dict(torch.load('learned_params/lstm.pt'))

	testX, testY = getData("data/observations_"+str(10*500)+".npy", "data/actions_"+str(10*500)+".npy")
	print(len(testX), len(testY))

	hnm_correct, ntm_correct, lstm_correct = 0, 0, 0
	totSamples = 0

	for i in range(50):

		print(i)

		s = torch.from_numpy(np.array(testX[i:i+1][0])).float().unsqueeze(0)
		s_lstm = s.view(s.size()[0], s.size()[2], -1)
		l = np.array(testY[i:i+1][0])

		(hnm_read_weights, hnm_write_weights) = hnm._initialise()
		(ntm_read_weights, ntm_write_weights) = ntm._initialise()
		lstm_h = lstm.h0.expand(s_lstm.size()[0],64)
		lstm_c = lstm.c0.expand(s_lstm.size()[0],64)

		for j in range(s.size()[1]):

			(hnm_out, hnm_read_weights, hnm_write_weights) = hnm.forward(s[:,j,:], hnm_read_weights, hnm_write_weights)
			(ntm_out, ntm_read_weights, ntm_write_weights) = ntm.forward(s[:,j,:], ntm_read_weights, ntm_write_weights)
			lstm_h, lstm_c, lstm_out = lstm.forward(s_lstm[:,:,j], lstm_h, lstm_c)

			if np.argmax(hnm_out.detach().numpy()) == np.argmax(l[j]): hnm_correct += 1
			if np.argmax(ntm_out.detach().numpy()) == np.argmax(l[j]): ntm_correct += 1
			if np.argmax(lstm_out.detach().numpy()) == np.argmax(l[j]): lstm_correct += 1

			totSamples += 1

	print(hnm_correct, ntm_correct, lstm_correct)
	print(totSamples)

def test():
	obstacle, wall_cw, wall_awc = Obstacle(), WallCW(), WallACW()
	obstacle_params, wall_cw_params, wall_acw_params = torch.load('program_memory/move.pt'), torch.load('program_memory/cw.pt'), torch.load('program_memory/acw.pt')

	networks = [obstacle, wall_cw, wall_awc]
	params = [obstacle_params, wall_cw_params, wall_acw_params]

	hnm = HNM(10, 14, networks, params)
	hnm.load_state_dict(torch.load('learned_params/hnm.pt'))

	testX, testY = getTestData()
	s = torch.from_numpy(np.array(testX[108:109][0])).float().unsqueeze(0)
	l = np.array(testY[108:109][0])

	print(s.size())
	# print(l.size())

	(read_weights, write_weights) = hnm._initialise()

	plt.matshow(hnm.Memory.detach().numpy())
	plt.show()

	correct = 0

	for i in range(s.size()[1]):
		(out, read_weights, write_weights) = hnm.forward(s[:,i,:], read_weights, write_weights)
		values = out.detach().numpy()
		if np.argmax(values) == np.argmax(l[i]): correct += 1 
		plt.matshow(hnm.Memory.detach().numpy())
		plt.show()

	print(correct)

def menu():

	print("-------- Menu for epuck avoid and follow task --------")
	print("What would you like to do today?")
	print("1. Train harvard machine")
	print("2. Train NTM")
	print("3. Train LSTM")
	print("4. Compare harvard machine, NTM, LSTM")
	print("5. Print memory trace and pointer distribution for harvard machine")
	print("0. Exit")

def main():

	menu()
	x = 10

	while x > 0:
		x = int(input())

		if x == 1:
			trainHNM()
		elif x == 2:
			trainNTM()
		elif x == 3:
			trainLSTM()
		elif x == 4:
			compare()
		elif x == 5:
			test()
		elif x == 0:
			break
		else:
			print("incorrect choice")

		menu()

	print("program exited")

main()

