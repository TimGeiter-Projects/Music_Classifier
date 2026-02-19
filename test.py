import pandas as pd

features = pd.read_csv("fma_metadata/features.csv", index_col=0)
k=1
for i in features.columns:
    print(k," : " ,i)
    k+=1
