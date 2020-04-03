import numpy as np

def getData():

	observations_1 = np.load("data/sequences_1.npy")
	actions_1 = np.load("data/action_1.npy")
	observations_2 = np.load("data/sequences_2.npy")
	actions_2 = np.load("data/action_2.npy")

	observations = np.concatenate((observations_1[:3000], observations_2[:1800]))
	actions = np.concatenate((actions_1[:3000], actions_2[:1800]))

	return observations, actions

def getTestData():

	observations = np.load("data/sequences_2.npy")
	actions = np.load("data/action_2.npy")

	observations = observations[1800:]
	actions = actions[1800:]

	return observations, actions


