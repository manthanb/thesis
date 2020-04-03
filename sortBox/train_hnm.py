import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from harvard_machine import HNM
from nets import Pick, RedBin, GreenBin

pick, redBin, greenBin = Pick(), RedBin(), GreenBin()
pickParams, redBinParams, greenBinParams = torch.load('program_memory/pick_box.pt'), torch.load('program_memory/navigate_to_red.pt'), torch.load('program_memory/navigate_to_green.pt'),

networks = [pick, redBin, greenBin]
params = [pickParams, redBinParams, greenBinParams]
featureExtractorParams = torch.load("learned_params/redOrGreen.pt")

hnm = HNM(10, 12, networks, params, featureExtractorParams)

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
		actionSequence = np.load("sequences/label/labelSequence_"+str(start)+".npy")
		
		imageSequence = torch.from_numpy(imageSequence).float()
		robotGpsSequence = torch.from_numpy(robotGpsSequence).float()
		y = torch.from_numpy(actionSequence).float()

		loss = hnm.train(imageSequence, y, robotGpsSequence, learning_rate)
		losses.append(loss.detach().numpy())
		
		print(i, inEpochCtr, loss)

		ctr += 1

	torch.save(hnm.state_dict(), "hnm.pt")
	np.save("losses/hnm", np.array(losses))
