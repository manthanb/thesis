import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.autograd import Variable

from harvard_machine import HNM
from lstm import LSTM
from ntm import NTM
from tasks import Tasks

def trainHarvard():
	t1 = Tasks()
	x_train, y_train = t1.sequence_type_1(2000)

	add_params, mul_params = torch.load('program_memory/add.pt'), torch.load('program_memory/mul.pt')
	hnm = HNM(10, 20, add_params, mul_params)

	hnm.train(x_train, y_train, 1)

def trainNTM():
	t = Tasks()
	x_train, y_train = t.sequence_type_1(2000)

	ntm = NTM(10, 20)

	ntm.train(x_train, y_train, 1, maxEpoch=25, learning_rate=0.0006)

def trainLSTM():
	t = Tasks()
	x_train, y_train = t.sequence_type_1(2000)

	lstm = LSTM(14, 256, 325, 1)
	lstm.train(x_train, y_train, maxEpoch=25, learning_rate=0.0003)

def compareFixed():
	t = Tasks()
	x_test, y_test = t.sequence_type_1(100)

	add_params, mul_params = torch.load('program_memory/add.pt'), torch.load('program_memory/mul.pt')
	hnm = HNM(10, 20, add_params, mul_params)
	hnm.load_state_dict(torch.load("learned_params/hnm_arch_2.pt"))

	ntm = NTM(10, 20)
	ntm.load_state_dict(torch.load("learned_params/ntm.pt"))

	lstm = LSTM(14, 256, 325, 1)
	lstm.load_state_dict(torch.load("learned_params/lstm.pt"))

	hnm_diff, lstm_diff, ntm_diff = 0, 0, 0

	for i in range(len(x_test)):
		hnm_out = hnm.recurrent_forward(x_test[i:i+1])
		ntm_out = ntm.recurrent_forward(x_test[i:i+1])
		lstm_out = lstm.recurrent_forward(x_test[i:i+1])

		answer = np.argmax(y_test[i:i+1].detach().numpy())
		hnm_diff += abs(answer - np.argmax(hnm_out.detach().numpy()))
		ntm_diff += abs(answer - np.argmax(ntm_out.detach().numpy()))
		lstm_diff += abs(answer - np.argmax(lstm_out.detach().numpy()))

	print(hnm_diff/len(y_test), ntm_diff/len(y_test), lstm_diff/len(y_test))


def menu():

	print("-------- Menu for math tasks --------")
	print("What would you like to do today?")
	print("1. Train harvard machine")
	print("2. Train NTM")
	print("3. Train LSTM")
	print("4. Compare harvard machine, NTM, LSTM on fixed length sequences")
	print("0. Exit")

def main():

	x = 10

	while x > 0:
		menu()
		x = int(input())

		if x == 1:
			trainHarvard()
		elif x == 2:
			trainNTM()
		elif x == 3:
			trainLSTM()
		elif x == 4:
			compareFixed()
		elif x == 5:
			compareVariable()
		elif x == 6:
			test()
		elif x == 0:
			break
		else:
			print("incorrect choice")

	print("program exited")

main()