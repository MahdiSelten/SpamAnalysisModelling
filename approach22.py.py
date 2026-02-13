#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("postdatav1.csv")


# In[3]:


data.sample(10)


# In[4]:


data.sample(20)


# In[ ]:





# In[5]:


import ast


# In[6]:


listofrows = []
for row in data["text"]:
    row = ast.literal_eval(row)
    listofrows.append(row)


# In[ ]:





# In[7]:


import gensim


# In[8]:


from gensim.models import word2vec


# In[9]:


from gensim.models import Word2Vec


# In[10]:


model = Word2Vec(sentences=listofrows, window=5, epochs=10, min_count=2, sg=0)


# In[11]:


model.wv["subject"]


# In[12]:


from tensorflow import keras


# In[13]:


from keras.layers import LSTM, Embedding, Dense


# In[14]:


RNNmodel = keras.Sequential()


# In[15]:


vocab_size = len(model.wv.key_to_index.items())


# In[16]:


vocab_size


# In[17]:


embedding_dim = model.vector_size


# In[18]:


embedding_dim


# In[19]:


import numpy as np


# In[20]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)


# In[21]:


embedding_matrix = np.zeros((len(tokenizer.word_index)+1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in model.wv:
        embedding_matrix[i] = model.wv[word]


# In[22]:


embedding_matrix


# In[48]:


RNNmodel.add(Embedding(input_dim=vocab_size, 
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True,
    ))


# In[49]:


RNNmodel.add(LSTM(32))


# In[50]:


RNNmodel.add(Dense(1, activation="sigmoid"))


# In[47]:


RNNmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[28]:


X = data["text"]


# In[29]:


y = data["label_num"]


# In[ ]:





# In[30]:


X_train, X_test, y_train, y_test = train_test_split(listofrows, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:





# In[31]:


def preprocesspad_sequencer(data, data2, tokenizer):
    texts = [" ".join(seq) for seq in data]
    textstest = [" ".join(seq) for seq in data2]

    totaltexts = texts+textstest
    totaltexts.extend(textstest)
    tokenizer.fit_on_texts(totaltexts)
    X_train_seq = tokenizer.texts_to_sequences(texts)
    X_test_seq = tokenizer.texts_to_sequences(textstest)

    return tokenizer, X_train_seq, X_test_seq



# In[ ]:





# In[33]:


tokenizer, X_train_seq, X_test_seq = preprocesspad_sequencer(X_train, X_test, tokenizer)


# In[34]:


from keras.utils import pad_sequences


# In[35]:


X_train_padded = pad_sequences(X_train_seq, maxlen=100, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding='post', truncating='post')


# In[36]:


print(X_train_padded[0][:10])


# In[37]:


from sklearn.utils import class_weight


# In[38]:


class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


# In[39]:


class_weights


# In[40]:


class_weights = dict(enumerate(class_weights))


# In[41]:


class_weights


# In[42]:


history = RNNmodel.fit(
    X_train_padded, 
    y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2,
    class_weight=class_weights

)


# In[ ]:


y_pred = RNNmodel.predict(X_test_padded)


# In[ ]:


y_pred_classes = (y_pred > 0.5).astype("int32")


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, y_pred_classes))


# In[ ]:


def preprocesspad_sequencer_input(userinput, tokenizer):


    userinput_seq = tokenizer.texts_to_sequences(userinput)


    user_input_padded = pad_sequences(userinput_seq, maxlen=100, padding='post', truncating='post')

    return user_input_padded



# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize


# In[ ]:


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





# In[ ]:


classiii, probclassi = preprocessing_input_predict(lemmatizer, stemmer)


# In[ ]:


probclassi


# In[ ]:


classiii


# In[ ]:


train_loss, train_acc = RNNmodel.evaluate(X_train_padded, y_train)


# In[ ]:


train_acc


# In[ ]:


print(history.history['loss'][-1], history.history['accuracy'][-1])
print(history.history['val_loss'][-1], history.history['val_accuracy'][-1])


# In[ ]:


data["label_num"].value_counts()


# In[ ]:




