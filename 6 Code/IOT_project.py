import pandas as pd
import re
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from openpyxl import Workbook
import LDAprocess
import Utils



# data load

data_2016 = pd.read_csv('../1 Data acquisition/2016.csv', encoding='cp949', header=None, usecols=[0])
data_2017_1 = pd.read_csv('../1 Data acquisition/2017_상.csv', encoding='cp949', header=None, usecols=[0])
data_2017_2 = pd.read_csv('../1 Data acquisition/2017_중.csv', encoding='cp949', header=None, usecols=[0])
data_2017_3 = pd.read_csv('../1 Data acquisition/2017_하.csv', encoding='cp949', header=None, usecols=[0])
data_2018_1 = pd.read_csv('../1 Data acquisition/2018_상.csv', encoding='cp949', header=None, usecols=[0])
data_2018_2 = pd.read_csv('../1 Data acquisition/2018_중.csv', encoding='cp949', header=None, usecols=[0])
raw_data = pd.concat([data_2016, data_2017_1, data_2017_2, data_2017_3, data_2018_1, data_2018_2])
raw_data = raw_data.to_numpy().reshape(raw_data.shape[0])



# preprocessing

# 특수기호와 숫자 제거
corpus = list(map(lambda s: re.sub("[^\sa-zA-Z가-힣]","",s), raw_data))

# stop words 제거
with open('IOT_stoplist.txt', 'r', encoding='cp949') as file:
    stopwords = file.readlines()
    stopwords = [s.strip() for s in stopwords]
    stopwords = '|'.join(stopwords)
corpus = list(map(lambda s: re.sub(stopwords, '', s), corpus))

# 명사 추출
engine = Kkma()
corpus = list(map(lambda s: engine.nouns(s), corpus))

# 단일문자 제거
for doc in corpus:
    for i, v in enumerate(doc):
        if len(v) == 1:
            del doc[i]

for doc in corpus:
    doc = ' '.join(doc)

# Term-Frequency matrix
# 빈도의 범위를 조절해 약 10,000 개의 term을 대상으로 matrix 구성함
CV = CountVectorizer(max_df=0.05, min_df=5)
term_document_matrix = CV.fit_transform(corpus)
feature_names = CV.get_feature_names()

# tf-idf value
# 특정 단어가 문서군에서 얼마나 중요한 것인지를 나타냄
TT = TfidfTransformer()
tfidf = TT.fit_transform(term_document_matrix)

wb = Workbook()
ws1 = wb.active
k = 0
col1 = 'A'  # document idx
col2 = 'B'  # term
col3 = 'C'  # tfidf value
B = tfidf.toarray()

for i in range(len(corpus)):
    for j in range(len(feature_names)):
        if B[i][j] > 0:
            k = k + 1
            col1 = 'A'
            col2 = 'B'
            col3 = 'C'
            col1 = col1 + str(k)
            col2 = col2 + str(k)
            col3 = col3 + str(k)
            ws1[col1] = i
            ws1[col2] = feature_names[j]
            ws1[col3] = B[i][j]
wb.save("C:TF_Matrix.xlsx")



# data modeling

# edge list data load & generate the doc-term tfidf matrix
csvfile = pd.read_csv(filename = "TFIDF_edgelist_0.05_5.csv", delimiter = ',')
doc_term_matrix = Utils.Edgelist2Matrix(csvfile)
doc_term_matrix.to_csv("DocTerm.csv", index=True)

# 최적 토픽수 결정
# elbow point method를 사용하여 결과적으로 33개를 최적 토픽수로 설정
TopicNumFrom = 2
TopicNumTo = 100
StepSize = 1
CosineOrPerplexity = 2
result = LDAprocess.TopicDecision(doc_term_matrix, TopicNumFrom, TopicNumTo, StepSize, CosineOrPerplexity)
result.to_csv("TopicDecision.csv", index=False)

# topic modeling
NumOfTopics = 33
DocTopic, TopicWord = LDAprocess.LDAexecute(doc_term_matrix, NumOfTopics)
TopicInfo = LDAprocess.TopicInfo(TopicWord)
DocTopic.to_csv("DocTopic.csv", index=False)
TopicWord.to_csv("TopicWord.csv", index=False)
TopicInfo.to_csv("TopicInfo.csv", index=False)
