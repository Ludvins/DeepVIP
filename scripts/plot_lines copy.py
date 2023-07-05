
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
os.chdir(".")

res1 = [i for i in glob.glob('results/LLA_*_MNIST_OOD*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
LLA = pd.concat(df)


res1 = [i for i in glob.glob('results/ELLA_MNIST_OOD*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
ELLA = pd.concat(df)


res1 = [i for i in glob.glob('results/VaLLA_MNIST_OOD*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA = pd.concat(df)



res1 = [i for i in glob.glob('results/VaLLA_Indep_MNIST_OOD*')]
df = []
for f in res1:
    try: 
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_I = pd.concat(df)

def std(x):
    return np.std(x)/np.sqrt(len(x))

VaLLA = VaLLA.astype({'M':'int'})
VaLLA_I = VaLLA_I.astype({'M':'int'})


ELLA = ELLA.astype({'M':'int'})
plt.rcParams['pdf.fonttype'] = 42

VaLLA["model"] = "VaLLA"
VaLLA_I["model"] = "VaLLA_I"

ELLA["model"] = "ELLA"
LLA["model"] = "LLA"

print(LLA)

df = pd.concat([LLA, VaLLA, VaLLA_I, ELLA]).drop(["Classes", "LOSS", "iterations", "weight_decay", "MAP_iterations", "K"], axis = 1)
print(df.groupby(["model",  "prior_std", "M", "Hessian", "Subset"], dropna=False).mean())
