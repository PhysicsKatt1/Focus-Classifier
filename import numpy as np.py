import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# p = [0, 0]
# log_worth= [41.557, 37.164]
# param= ['N Dense Layers', 'N Conv2D Layers']
p = [0.06100, 0.18016, 0.18138]
log_worth = [1.215, 0.744, 0.741]
param = ['N Filters Dense 1', 'N Filters Conv2D 1', 'N Filters Dense 1 * N Filters Conv2D 1']

df = pd.DataFrame({'P Value': p, 'Log Worth':log_worth, 'Parameter':param})

plt.figure(figsize=(12,8))
sns.barplot(data = df, x = 'Log Worth', y = 'Parameter', hue = 'Parameter', palette='cool')
plt.suptitle('Response Validation Accuracy')
plt.savefig( r'/Users/trentstarkey/Desktop/Classifier_AVOVA_val.png')