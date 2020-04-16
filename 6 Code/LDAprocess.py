import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def LDAexecute(DocWord, NumOfTopics, TopicWordNomalization = True):
    LDA = LatentDirichletAllocation(n_components=NumOfTopics, learning_method='batch', evaluate_every=1, n_jobs=1, verbose = 1)

    data = LDA.fit_transform(DocWord)
    topic_names = ["Topic%i" % i for i in range(1, NumOfTopics + 1)]
    DocTopic = pd.DataFrame(data, columns=topic_names, index=DocWord.index)
    if TopicWordNomalization:
        TopicWord = pd.DataFrame(LDA.components_ / LDA.components_.sum(axis=1)[:, np.newaxis], columns=DocWord.columns, index=topic_names)
    else:
        TopicWord = pd.DataFrame(LDA.components_, columns=DocWord.columns, index=topic_names)
    return DocTopic, TopicWord



def TopicInfo(TopicWord):
    list_temp = []
    col_name = []
    for _, row in TopicWord.iterrows():
        temp = row.sort_values(ascending=False).iloc[0:20]   # 상위 20개의 term만 취함
        temp_row = []
        for i in range(20):
            temp_row.append(temp.index[i])
            temp_row.append(temp.iloc[i])
            col_name.append('%d_Word' % i)
            col_name.append('%d_Prob' % i)
        list_temp.append(temp_row)
    TopicInfo = pd.DataFrame(data=list_temp, columns=col_name, index=TopicWord.index)
    return TopicInfo



def TopicDecision(DocWord, TopicNumFrom = 1, TopicNumTo = 10, StepSize=1, CosineOrPerplexity = 0):
    """Return DataFrame of cosine similarity or perplexity"""
    LDAprocesses = []
    TopicNum = [TopicNumFrom]
    if StepSize == 1:
        TopicNum = list(range(TopicNumFrom, TopicNumTo+1))
    else:
        temp = TopicNumFrom
        while temp < TopicNumTo:
            temp = temp + StepSize
            TopicNum.append(temp)
    for i in TopicNum:
        print("Processing NumOfTopic: %i"%i)
        LDAprocesses.append(LatentDirichletAllocation(n_components=i, learning_method='batch', max_iter=10).fit(DocWord))
    if CosineOrPerplexity == 0:
        list_similarity = []
        for i in LDAprocesses:
            temp_TopicWord = i.components_ / i.components_.sum(axis=1)[:, np.newaxis]
            list_similarity.append(np.average(cosine_similarity(X = temp_TopicWord, Y = None)))
        plt.plot(TopicNum, list_similarity)
        plt.ylabel('Avg cosine similarity')
        plt.xlabel('# of Topics')
        plt.show()
        return pd.DataFrame(list_similarity, index=TopicNum, columns=["avg cosine sim"])
    elif CosineOrPerplexity == 1:
        list_perplexity = []
        for i in LDAprocesses:
            list_perplexity.append(i.perplexity(DocWord, sub_sampling=True))
        plt.plot(TopicNum, list_perplexity)
        plt.ylabel('Perplexity')
        plt.xlabel('# of Topics')
        plt.show()
        return pd.DataFrame(list_perplexity, index=TopicNum, columns=["perplexity"])
    elif CosineOrPerplexity == 2:
        list_similarity = []
        for i in LDAprocesses:
            temp_TopicWord = i.components_ / i.components_.sum(axis=1)[:, np.newaxis]
            list_similarity.append(np.average(cosine_similarity(X = temp_TopicWord, Y = None)))
        list_perplexity = []
        for i in LDAprocesses:
            list_perplexity.append(i.perplexity(DocWord, sub_sampling=True))
        return pd.concat([pd.DataFrame(list_similarity, index=TopicNum, columns=["avg cosine sim"]), pd.DataFrame(list_perplexity, index=TopicNum, columns=["perplexity"])], axis=1)
