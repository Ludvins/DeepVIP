import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd

os.chdir(".")
res1 = [i for i in glob.glob("results/MAP_dataset=*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
MAP = pd.concat(df)

res1 = [i for i in glob.glob("results/MAP_Conv_dataset=*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
MAP_conv = pd.concat(df)

res1 = [i for i in glob.glob("results/LLA_dataset=*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
LLA = pd.concat(df)


res1 = [i for i in glob.glob("results/ELLA_dataset=*")]
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

res1 = [i for i in glob.glob("results/VaLLA_MC_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_MC = pd.concat(df)

res1 = [i for i in glob.glob("results/VaLLA_RBF_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_RBF = pd.concat(df)

res1 = [i for i in glob.glob("results/VaLLA_Conv_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_Conv = pd.concat(df)

res1 = [i for i in glob.glob("results/VaLLA_Conv_MC_dataset*")]
df = []
for f in res1:
    try:
        df.append(pd.read_csv(f))
    except:
        continue
VaLLA_Conv_MC = pd.concat(df)


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
VaLLA_Conv["model"] = "VaLLA RBF Conv"
VaLLA_Conv_MC["model"] = "VaLLA RBF Conv MC"
VaLLA_MC["model"] = "VaLLA MC"
MAP["model"] = "MAP"
MAP_conv["model"] = "MAP CNN"
# VaLLA_I["model"] = "VaLLA_I"

ELLA["model"] = "ELLA"
LLA["model"] = "LLA"

df = pd.concat([MAP, MAP_conv, LLA, ELLA, VaLLA, VaLLA_RBF, VaLLA_MC, VaLLA_Conv, VaLLA_Conv_MC]).drop(
    ["LOSS", "Unnamed: 0", "iterations", "weight_decay", "MAP_iterations", "CRPS"], axis=1
)

regression = df[df["ACC"].isna()].drop(["ACC",  "ECE",  "NLL MC",  "ACC MC",  "ECE MC"], axis = 1)
classification = df[df["RMSE"].isna()].drop(["RMSE",  "log_variance"], axis = 1)

# df = df.astype({"M": "int"})
print(regression.groupby(["dataset", "model","M", "subset", "hessian"], dropna=False).mean())
print(classification.groupby(["dataset", "model","M", "subset", "hessian"], dropna=False).mean())
