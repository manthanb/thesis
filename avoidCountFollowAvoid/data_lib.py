import numpy as np

def getData(observations_file, actions_file):

	observations= np.load(observations_file)
	actions = np.load(actions_file)
	return observations, actions
