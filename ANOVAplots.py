import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

p = [0.000223, 0.00557]
log_worth = [3.652, 2.254]
param = ['Secondary Val Split', 'LR']

df = pd.DataFrame({'P Value': p, 'Log Worth':log_worth, 'Parameter':param})

plt.figure(figsize=(12,8))
sns.barplot(data = df, x = 'Log Worth', y = 'Parameter', hue = 'Parameter', palette='cool')
plt.suptitle('Response Training Accuracy')
plt.savefig( r'/Users/trentstarkey/Desktop/Classifier_AVOVA_train.png')