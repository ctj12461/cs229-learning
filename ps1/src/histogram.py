import matplotlib.pyplot as plt
import numpy as np

import util

ds, y = util.load_dataset("../data/ds4_train.csv")
feat1 = ds[:, 2]
feat2 = ds[:, 3]

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].hist(feat1, bins=50)
axs[1].hist(feat2, bins=50)
axs[2].hist(y, bins=50)
plt.show()