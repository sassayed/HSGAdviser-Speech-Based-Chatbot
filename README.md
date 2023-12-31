# Speech-based Chatbot

**Welcome to Speech-Chatbot!** This is an dvising speech-based chatbot developed by using a deep learning model to support high school students. The dataset in this study is created from 1791 pairs of questions and answers. In this model, we introduce an embedding layer alongside the LSTM layer, followed by the integration of a single dense layer with a Softmax function to produce the response.


**This Seq2Seq mode is developed as following:**

**1- Importing the required libraries from tensorflow**

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

**2-Loading the dataset Pairs of questions and answers**

**3- Preprocessing the dataset**

**4- Creating the Dictionary**

**5- Padding the sequences**

**6- Building the Model by using the below layers:**

**- Embedding Layer**

```python
embed = Embedding(VOCAB_SIZE+1, output_dim=50, 
                  input_length=13,
                  trainable=True                  
                  )
```
**- LSTM Layer**
```python
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
```
**- Dense Layer**
```python
dense = Dense(VOCAB_SIZE, activation='softmax')
```
