import pandas as pd

dfFullData = pd.read_json("train_codesearchnet_7.json")

dfLabel = dfFullData["label"]
print(dfLabel.head())
fullLength = len(dfFullData["label"])
numZeros = (dfFullData["label"] == 0).sum()
print("Fraction of zeros in full data:", numZeros /
      fullLength, "length of full data is:", fullLength)

#print("NUM 0 in original {} NUM 1 IN ORIGINAL {}")
#print("NUM 0 in TRIMMED {} NUM 1 IN TRIMMED {}")


dfTrimmedData = pd.read_json("train_codesearchnet_7_trimmed.json")

fullLength = len(dfTrimmedData["label"])
numZeros = (dfTrimmedData["label"] == 0).sum()
print("Fraction of zeros in trimmed data:", numZeros /
      fullLength, "length of trimmed data is:", fullLength)
