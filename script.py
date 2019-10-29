import pandas as pd
import numpy as np
from numpy import isnan
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
from bs4 import BeautifulSoup
import sys
import codecs
from nltk.tokenize import word_tokenize
if sys.stdout.encoding != 'cp850':
    sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'cp850':
    sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

def initial_clean(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower()
    text=" ".join(text.split()) # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    return [word for word in text if word not in stop_words]

stemmer = PorterStemmer()
def stem_words(text):
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1]
    except IndexError:
        pass
    return text

def apply_all(text):
    return stem_words(remove_stop_words(initial_clean(text)))

def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]

def train_lda(data):
    num_topics = 100
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    return dictionary,corpus,lda

def jensen_shannon(query, matrix):
    p = query[None,:].T
    q = matrix.T
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=10):
    sims = jensen_shannon(query,matrix)
    return sims

from keywords import KeyWords
from nltk.corpus import stopwords

# with open('keyword-extract-master/5g.txt', 'r') as f:
#     data = f.read()
    #print(type(data))

df = pd.read_csv(sys.argv[1], header=None, names=["text"])
# df2 = pd.read_csv('full_abstractremoved.csv', header=None, names=["text"])
# df3 = pd.read_csv('shabby_abremoved.csv', header=None, names=["text"])
# df=pd.concat([df1,df2,df3])
test_df = pd.read_csv(sys.argv[2], header=None, names=["text"])
# test_df2=pd.read_csv('full_ab.csv', header=None, names=["text"])
# test_df3=pd.read_csv('abstractTextShahbaz.csv', header=None, names=["text"])
# test_df=pd.concat([test_df1,test_df2,test_df3])
# test_df.append(test_df2, ignore_index = True)
df.dropna(axis=0, inplace=True, subset=['text'])
test_df.dropna(axis=0, inplace=True, subset=['text'])
df['tokenized'] = df['text'].apply(apply_all)
test_df['tokenized'] = test_df['text'].apply(apply_all)
# print(df.head())
# print(test_df.head())
all_words = [word for item in list(df['tokenized']) for word in item]
fdist = FreqDist(all_words)
# print(len(fdist))
k=int(len(fdist)/2.8)
top_k_words = fdist.most_common(k)
# print(top_k_words[-10:])
top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
dfToList = df['text'].tolist()

final_list=[]
for i in range(len(dfToList)):
    if i%9==0:
        with open('testcorpus.txt', 'r', encoding="utf8") as f1:
            corpus_1 = f1.read()
        stopWords = stopwords.words('english')
        keyword = KeyWords(corpus=corpus_1, stop_words=stopWords, alpha=0.8)
        d = keyword.get_keywords(str(dfToList[i]), n=2)
        #final_list=[]
        for pair in d:
            for kw in word_tokenize(pair[0]):
                final_list.append(kw)


ps = PorterStemmer()

for i in range(len(final_list)):
    top_k_words.add(ps.stem(final_list[i]))

# print(len(top_k_words))
# print(type(top_k_words))
df['tokenized'] = df['tokenized'].apply(keep_top_k_words)
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

df = df[df['tokenized'].map(type) == list]

dictionary,corpus,lda = train_lda(df)
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])

 #j=0
finalans=[]
for i in range(test_df['text'].count()):
    new_bow = dictionary.doc2bow(test_df.iloc[i,1])
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])
    most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist)
    # j+=most_sim_ids[i]
    # np.nan_to_num(most_sim_ids)
    # a = array([[1, 2, 3], [0, 3, NaN]])
    where_are_NaNs = isnan(most_sim_ids)
    most_sim_ids[where_are_NaNs] = 0
    most_sim_ids=1-most_sim_ids
    #print(most_sim_ids)
    finalans.append(most_sim_ids)
#print(j)
final2d=np.array(finalans)
np.savetxt("similarity_matrix.csv", final2d, delimiter=",")
print("The answer lies in similarity_matrix.csv")
