import pandas as pd
from sklearn.utils import resample

dfFullData = pd.read_json("train_codesearchnet_7.json")
# print(dfFullData.head())
dfTrimmedData = resample(dfFullData, n_samples=len(
    dfFullData)//12, stratify=dfFullData["label"])
# print(dfTrimmedData.head())

# print(len(dfTrimmedData))
# print(len(dfFullData))
# orient="records" is so the json format is correct
dfTrimmedData.to_json('./train_codesearchnet_7_trimmed.json', orient="records")
