
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
os.chdir(".")


res1 = [i for i in glob.glob('results/ELLA_dataset=Spiral3*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
df = pd.concat(df)
ELLA = df.loc[:, (df != df.iloc[0]).any()] 


res1 = [i for i in glob.glob('results/VaLLA_dataset=Spiral3*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
df = pd.concat(df)
VaLLA = df.loc[:, (df != df.iloc[0]).any()] 


def std(x):
    return np.std(x)/np.sqrt(len(x))

VaLLA = VaLLA.astype({'M':'int'})
ELLA = ELLA.astype({'M':'int'})
plt.rcParams['pdf.fonttype'] = 42

VaLLA["model"] = "VaLLA"
ELLA["model"] = "ELLA"

df = pd.concat([VaLLA, ELLA])

print(df)
print(df.groupby(["model", "M"]).mean())
input()
lm = sns.catplot(data = df, x = "M", y="KL", palette="YlGnBu_d", hue="model", errorbar="se", kind="point", height=6, aspect=.75, legend = False)
lm.fig.set_size_inches(16,10)

ax = lm.axes[0][0]  # access a grid of 'axes' objects
ax.set_facecolor('white')
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel("Inducing points | Nyström features", fontsize=26)
ax.set_ylabel("Kullback-Leibler Divergence", fontsize=26)
#axis[2].xaxis.set_tick_params(labelsize=20)
#axis[2].yaxis.set_tick_params(labelsize=20)
ax.legend(prop={'size': 22})
plt.savefig("KL.pdf", bbox_inches='tight')
plt.clf()




lm = sns.catplot(data = df, x = "M", y="MAE", palette="YlGnBu_d", hue="model", errorbar="se", kind="point", height=6, aspect=.75, legend = False)
lm.fig.set_size_inches(16,10)

ax = lm.axes[0][0]  # access a grid of 'axes' objects
ax.set_facecolor('white')
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel("Inducing points | Nyström features", fontsize=26)
ax.set_ylabel("Mean Absolute Error", fontsize=26)
#axis[2].xaxis.set_tick_params(labelsize=20)
#axis[2].yaxis.set_tick_params(labelsize=20)
ax.legend(prop={'size': 22})
plt.savefig("MAE.pdf", bbox_inches='tight')
plt.clf()

