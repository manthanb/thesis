import torch
import random
import numpy as np

class Tasks:

	def _one_hot(self, symbol, length):
		res = np.zeros((length,))

		if symbol == "+":
			res[10] = 1
		elif symbol == "*":
			res[11] = 1
		elif symbol == "(":
			res[12] = 1
		elif symbol == ")":
			res[13] = 1
		elif symbol == "$":
			res[0] = 1
		else:
			res[symbol] = 1

		return res

	def sequence_type_1(self, num_train_samples):

		x_train, y_train = [], []

		for i in range(num_train_samples):

			operand1 = np.squeeze(random.sample(range(1,10),1))
			operand2 = np.squeeze(random.sample(range(1,10),1))
			operand3 = np.squeeze(random.sample(range(1,10),1))
			operand4 = np.squeeze(random.sample(range(1,10),1))

			result = (operand1+operand2) * (operand3+operand4)		
			input1, input2 = self._one_hot(operand1,14), self._one_hot(operand2,14)
			symbol1, symbol2 = self._one_hot("+", 14), self._one_hot("*", 14)
			input3, input4 = self._one_hot(operand3,14), self._one_hot(operand4,14)
			bopen, bclose = self._one_hot("(", 14), self._one_hot(")", 14)
			dollar = self._one_hot("$", 14)
			output = self._one_hot(result, 325)

			ip = np.concatenate(([bopen],[input1],[symbol1],[input2],[bclose],[symbol2],[bopen],[input3],[symbol1],[input4],[bclose],[dollar]))
			x_train.append(ip.T)
			y_train.append([output])

		return torch.from_numpy(np.array(x_train)).float(), torch.from_numpy(np.array(y_train)).float()
