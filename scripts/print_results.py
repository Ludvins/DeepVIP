import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd

os.chdir(".")
res1 = [i for i in glob.glob("results/MAP_*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
MAP = pd.concat(df)


res1 = [i for i in glob.glob("results/LLA_*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
LLA = pd.concat(df)


res1 = [i for i in glob.glob("results/ELLA_*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
ELLA = pd.concat(df)


res1 = [i for i in glob.glob("results/VaLLA_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA = pd.concat(df)

res1 = [i for i in glob.glob("results/VaLLA_RBF_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_RBF = pd.concat(df)



# res1 = [i for i in glob.glob("results/VaLLA_Indep_MNIST_OOD*")]
# df = []
# for f in res1:
#     try:
#         df.append(pd.read_csv(f))
#     except:
#         continue
# VaLLA_I = pd.concat(df)


def std(x):
    return np.std(x) / np.sqrt(len(x))


VaLLA["model"] = "VaLLA"
VaLLA_RBF["model"] = "VaLLA RBF"
MAP["model"] = "MAP"
# VaLLA_I["model"] = "VaLLA_I"

ELLA["model"] = "ELLA"
LLA["model"] = "LLA"

print(LLA)

df = pd.concat([MAP, LLA, ELLA, VaLLA, VaLLA_RBF]).drop(
    ["LOSS", "Unnamed: 0", "iterations", "weight_decay", "MAP_iterations"], axis=1
)


# df = df.astype({"M": "int"})
print(df.groupby(["model","M", "subset", "hessian"], dropna=False).mean())
