#!/usr/bin/env python
# coding: utf-8

# # Story Generator

# ## Importing libraries

# In[3]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
import tensorflow as tf
import pickle


# ## Analyzing dataset

# In[4]:


data=open('stories.txt',encoding="utf8").read()


# In[5]:


#data


# ## NLP

# In[6]:


# Converting the text to lowercase and splitting it
corpus = data.lower().split("\n")


# In[7]:


#corpus


# In[6]:


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(total_words)


# In[ ]:


pickle.dump(tokenizer,open('transform.pkl','wb'))


# In[ ]:


# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# In[8]:


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


# In[ ]:


# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]


# In[ ]:


label = ku.to_categorical(label, num_classes=total_words)


# ## Model building

# In[11]:


model = Sequential()
model.add(Embedding(total_words, 300, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(200, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


history = model.fit(predictors, label, epochs=200, verbose=0)


# ## Graphs

# In[13]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()


# ## Saving the model

# In[14]:


model.save('model_final.h5')
print("Saved model to disk")


# In[ ]:


# serialize model to JSON
model_json=model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


get_ipython().system('pip install pickle-mixin')


# In[ ]:


weigh= loaded_model.get_weights(); 
pklfile= "model.pkl"
fpkl= open(pklfile, 'wb')    #Python 3     
pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
fpkl.close()


# ## Loading the trained model (if required)

# In[ ]:


#!pip install h5py


# In[ ]:


# Load json and create model
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=tf.keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[ ]:


#loaded_model.summary()


# ## Prediction

# In[16]:


seed_text = "As i walked, my heart sank"
next_words = 100
  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)


# In[ ]:




