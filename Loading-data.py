import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# load the datase
df = pd.read_csv(r"C:\Users\MOPA\ML-1\Chatbot\Speech\QA.csv", encoding_errors='ignore')
