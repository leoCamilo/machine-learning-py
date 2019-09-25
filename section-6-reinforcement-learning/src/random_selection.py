import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../data/Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()