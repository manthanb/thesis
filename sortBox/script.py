import numpy as np
import random
import matplotlib.pyplot as plt

losses_ntm = np.load("losses/ntm.npy")
losses_hnm = np.load("losses/hnm.npy")


X = np.array([x for x in range(1,41)])

ntmLosses, hnmLosses = [], []
i = 0

while i+60 < len(losses_hnm):
	ntmLosses.append(sum(losses_ntm[i:i+60])/60)
	hnmLosses.append(sum(losses_hnm[i:i+60])/60)
	i += 60

plt.plot(X, hnmLosses, label="Harvad Machine")
plt.plot(X, ntmLosses, label="Neural Turing Machine")

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")

plt.legend()
plt.show()

