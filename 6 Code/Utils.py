import pandas as pd
import numpy as np



def Edgelist2Matrix(edgelist):
    # dataframe 값의 중복을 제거하여 matrix로 변환
    source = np.unique(edgelist[edgelist.columns[0]].values)
    target = np.unique(edgelist[edgelist.columns[1]].values)
    matrix = pd.DataFrame(np.zeros([len(source), len(target)]), index=source, columns=target)
    for _, row in edgelist.iterrows():
        Source, Target, Weight = row
        try:
            matrix.loc[Source, Target] = Weight
        except KeyError:
            continue
    return matrix



def Matrix2Edgelist(matrix):
    edgelist = []
    for source in matrix.index.values:
        for target in matrix.columns.values:
            edgelist.append((target,source,matrix.loc[source][target]))
    return pd.DataFrame(edgelist, columns=['Source','Target','Weight'])
