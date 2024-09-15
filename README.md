# Speech-based Chatbot

**Welcome to Speech-Chatbot!** This chatbot, designed for advising high school students, is powered by a deep learning model with speech-based interaction. The dataset used for this study consists of 1791 question-answer pairs. The model architecture includes an embedding layer, which feeds into an LSTM layer, and concludes with a single dense layer using a Softmax function to generate the chatbot's responses.


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
