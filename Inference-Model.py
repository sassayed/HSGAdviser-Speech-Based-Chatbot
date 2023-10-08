# Inference Model
import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np

enc_model = Model([enc_inp], enc_states)
from keras.preprocessing.sequence import pad_sequences
# decoder Model
decoder_state_input_h = Input(shape=(400*2,))
decoder_state_input_c = Input(shape=(400*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = dec_lstm(dec_embed , 
                                    initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
dec_model = Model([dec_inp]+ decoder_states_inputs,
                                      [decoder_outputs]+ decoder_states)
# Function to handle user input and generate responses (you should replace this with your chatbot logic)
def get_response():
  #  prepro1 = ""
  #  prepro1  = input("you : ")
    prepro1 = input_text.get("1.0", tk.END).strip()
    response_text.insert(tk.END, f"User: {prepro1}\n")
    response_text.delete("1.0", tk.END)
    # Replace this with your chatbot's response generation logic
  #    bot_response = "Chatbot: This is a sample response."

    prepro1 = clean_text(prepro1)

    prepro = [prepro1]

    txt = []
    for x in prepro:
        lst = []
        for y in x.split():
            try:
                lst.append(vocab[y])
                ## vocab['hello'] = 454
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)
    txt = pad_sequences(txt, 13, padding='post')
    stat = enc_model.predict( txt )
    empty_target_seq = np.zeros( ( 1 , 1) )
     ##   empty_target_seq = [0]
    empty_target_seq[0, 0] = vocab['<SOS>']
    ##    empty_target_seq = [255]
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h, c= dec_model.predict([ empty_target_seq] + stat )
        decoder_concat_input = dense(dec_outputs)
               sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
            sampled_word = inv_vocab[sampled_word_index] + ' '
        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word  
        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
            stop_condition = True 
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
         stat = [h, c]  
        response_text.insert(tk.END, f"{decoded_translation}\n")
   # input_text.delete("1.0", tk.END)
    # Create the Interface
root = tk.Tk()
root.title("High School Advising Chatbot Interface")
# Create a text box for user input
input_label = tk.Label(root, text="Student' enquiry:")
input_label.pack()
input_text = scrolledtext.ScrolledText(root, height=5, width=40)
input_text.pack()
# Create a text box for displaying responses
response_label = tk.Label(root, text="HS Advising Chatbot Response:")
response_label.pack()
response_text = scrolledtext.ScrolledText(root, height=10, width=40)
response_text.pack()
# Create a button to send user input and get a response
send_button = tk.Button(root, text="Send", command=get_response)
send_button.pack()
# Run the main loop
root.mainloop()
