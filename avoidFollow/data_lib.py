import numpy as np

def getData():

	observations = np.load("data/observations.npy")
	actions = np.load("data/actions.npy")

	print(actions.shape)
	print(observations.shape)

	return observations[:6800], actions[:6800]

def getTestData():

	observations = np.load("data/observations.npy")
	actions = np.load("data/actions.npy")

	print(actions.shape)
	print(observations.shape)

	return observations[6800:], actions[6800:]


