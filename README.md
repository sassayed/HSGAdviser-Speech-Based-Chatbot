# Speech-Based Chatbot

**Welcome to Speech-Chatbot!** This is an dvising speech-based chatbot developed by using a deep learning model to support high school students. The corpus in this study is created from 1791 pairs of questions and answers. The model is designed by using the encoder-decoder (Seq2Seq). An embedding layer is added to the LSTM layer and a single dense layer with Softmax function is connected in this model to generate the response.


**This mode is designed as following**
# 1- Loading the required libraries

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences```

