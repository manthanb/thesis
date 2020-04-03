import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from ntm import NTM

featureExtractorParams = torch.load("learned_params/redOrGreen.pt")
ntm = NTM(10, 12, featureExtractorParams)

epochs = 40
initial_learning_rate = 0.0006
learning_rate = 0.0006
ctr = 0
decay = 0.001
bit = 0
losses = []

for i in range(epochs):

	print("epoch:", i)

	sampleIdxRed = set([x for x in range(1,31)])
	sampleIdxGreen = set([x for x in range(31,61)])
	sampleIdx = None
	inEpochCtr = 0

	while len(sampleIdxGreen) > 0 or len(sampleIdxRed) > 0:

		learning_rate = initial_learning_rate * (1 / (1 + decay * ctr))
		inEpochCtr+=1

		if bit == 0:
			sampleIdx = sampleIdxRed
			bit = 1
		else:
			sampleIdx = sampleIdxGreen
			bit = 0

		start = random.sample(sampleIdx, 1)[0]
		sampleIdx.remove(start)

		print(start)

		imageSequence = np.load("sequences/image/imageSequence_"+str(start)+".npy")
		robotGpsSequence = np.load("sequences/robot/robotGpsSequence_"+str(start)+".npy")
		actionSequence = np.load("sequences/action/actionSequence_"+str(start)+".npy")
		
		imageSequence = torch.from_numpy(imageSequence).float()
		robotGpsSequence = torch.from_numpy(robotGpsSequence).float()
		y = torch.from_numpy(actionSequence).float()

		loss = ntm.train(imageSequence, y, robotGpsSequence, learning_rate)
		losses.append(loss.detach().numpy())
		
		print(i, inEpochCtr, loss)

		ctr += 1

	torch.save(ntm.state_dict(), "ntm.pt")
	np.save("losses/ntm", np.array(losses))
