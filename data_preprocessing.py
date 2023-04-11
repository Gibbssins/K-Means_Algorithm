import numpy as np
import pandas as pd

def data_preprocessing(data):

    if isinstance(data,pd.DataFrame):
        columns = data.columns
        df = data.select_dtypes(include=['number'])
        columns_number = df.columns
        df = df.to_numpy()
        columns_means = np.mean(df,axis=0, keepdims=True)
        columns_std = np.std(df,axis=0,keepdims=True)
        df = (df-columns_means)/columns_std
        df = pd.DataFrame({columns_number[i]: df[:,i] for i in range(len(columns_number))})
        df1 = data.select_dtypes(exclude=['number'])
        data = pd.concat([df,df1], axis=1)
        print(data.shape)
        return data
    else:
        columns_means = np.mean(data, axis=0, keepdims=True)
        columns_std = np.std(data, axis=0, keepdims=True)
        df = (data - columns_means) / columns_std
        return df