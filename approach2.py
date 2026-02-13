#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd


# In[65]:


data = pd.read_csv("postdatav1.csv")


# In[66]:


data.sample(10)


# In[67]:


data.sample(20)


# In[ ]:





# In[68]:


import ast


# In[69]:


listofrows = []
for row in data["text"]:
    row = ast.literal_eval(row)
    listofrows.append(row)


# In[ ]:





# In[70]:


import gensim


# In[71]:


from gensim.models import word2vec


# In[72]:


from gensim.models import Word2Vec


# In[73]:


model = Word2Vec(sentences=listofrows, window=5, epochs=10, min_count=2, sg=0)


# In[74]:


model.wv["subject"]


# In[75]:


from tensorflow import keras


# In[76]:


from keras.layers import LSTM, Embedding, Dense


# In[77]:


RNNmodel = keras.Sequential()


# In[78]:


vocab_size = len(model.wv.key_to_index.items())


# In[79]:


vocab_size


# In[80]:


embedding_dim = model.vector_size


# In[81]:


embedding_dim


# In[82]:


import numpy as np


# In[83]:


embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in model.wv.key_to_index.items():
    embedding_matrix[i] = model.wv[word]


# In[84]:


embedding_matrix


# In[ ]:


RNNmodel.add(Embedding(input_dim=vocab_size, 
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True,
    ))


# In[86]:


RNNmodel.add(LSTM(32, return_sequences=True))


# In[87]:


RNNmodel.add(LSTM(64))


# In[88]:


RNNmodel.add(Dense(1, activation="sigmoid"))


# In[89]:


RNNmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X = data["text"]


# In[92]:


y = data["label_num"]


# In[ ]:





# In[93]:


X_train, X_test, y_train, y_test = train_test_split(listofrows, y, test_size=0.2, random_state=42, stratify=y)


# In[94]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[95]:


def preprocesspad_sequencer(data, data2):
    texts = [" ".join(seq) for seq in data]
    textstest = [" ".join(seq) for seq in data2]

    tokenizer = Tokenizer(num_words=5000)
    totaltexts = texts+textstest
    totaltexts.extend(textstest)
    tokenizer.fit_on_texts(totaltexts)
    X_train_seq = tokenizer.texts_to_sequences(texts)
    X_test_seq = tokenizer.texts_to_sequences(textstest)

    return tokenizer, X_train_seq, X_test_seq



# In[ ]:





# In[96]:


tokenizer, X_train_seq, X_test_seq = preprocesspad_sequencer(X_train, X_test)


# In[97]:


from keras.utils import pad_sequences


# In[98]:


X_train_padded = pad_sequences(X_train_seq, maxlen=100, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding='post', truncating='post')


# In[99]:


print(X_train_padded[0][:10])


# In[100]:


from sklearn.utils import class_weight


# In[101]:


class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


# In[102]:


class_weights


# In[103]:


class_weights = dict(enumerate(class_weights))


# In[104]:


class_weights


# In[105]:


history = RNNmodel.fit(
    X_train_padded, 
    y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2,
    class_weight=class_weights

)


# In[106]:


y_pred = RNNmodel.predict(X_test_padded)


# In[107]:


y_pred_classes = (y_pred > 0.5).astype("int32")


# In[108]:


from sklearn.metrics import classification_report


# In[109]:


print(classification_report(y_test, y_pred_classes))


# In[110]:


def preprocesspad_sequencer_input(userinput, tokenizer):

    tokenizer = Tokenizer(num_words=5000)

    userinput_seq = tokenizer.texts_to_sequences(userinput)


    user_input_padded = pad_sequences(userinput_seq, maxlen=100, padding='post', truncating='post')

    return user_input_padded



# In[111]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize


# In[112]:


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
def preprocessing_input_predict(lemmatizer, stemmer):
    textinput = input("Give a message")

    tokenized_sentence = word_tokenize(textinput)

    lemmatized_words = [lemmatizer.lemmatize(w) for w in tokenized_sentence]
    stemmed_and_lemmatized = [stemmer.stem(w) for w in lemmatized_words]
    textinput = " ".join(stemmed_and_lemmatized)

    padded = preprocesspad_sequencer_input(textinput, tokenizer)

    prediction = RNNmodel.predict(padded)

    classified_prediction = (prediction > 0.5).astype("int32")
    proba_classified_prediction = prediction
    return classified_prediction, proba_classified_prediction





# In[113]:


classiii, probclassi = preprocessing_input_predict(lemmatizer, stemmer)


# In[114]:


probclassi


# In[115]:


classiii


# In[116]:


train_loss, train_acc = RNNmodel.evaluate(X_train_padded, y_train)


# In[117]:


train_acc


# In[118]:


print(history.history['loss'][-1], history.history['accuracy'][-1])
print(history.history['val_loss'][-1], history.history['val_accuracy'][-1])


# In[119]:


data["label_num"].value_counts()


# In[ ]:




