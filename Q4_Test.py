"""
PS#2
Q5 (Testing) - A Large Character Level LSTM
Loads a trained LSTM and mapping and generates sentences
Implemented from Q4 testing code
"""

import numpy as np
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = 'the new era began the king was tried doomed and beheaded'
n_chars_to_predict = 500
seq_length = 100

# load the model and mapping
model = load_model('LargeLSTM_model.h5')
mapping = load(open('LargeLSTM_mapping.pkl', 'rb'))


# Make predictions
for k in range(n_chars_to_predict):
    # encode the characters as integers
    encoded = [mapping[char] for char in seed_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    # encoded = to_categorical(encoded, num_classes=len(mapping))
    
    # Tried without converting len(mapping) to float, would only output "...mmmmmmmm..."
    encoded /= float(len(mapping))
    
    # numpy.reshape(a, newshape, order='C'
    # Reshapes the array so that it is easily processed because it is different from 
    # the Q4 model -- The model was coded differently so it needs to be parsed differently
    encoded = numpy.reshape(encoded,    #array to bereshaped
                            (encoded.shape[0],  #newshape
                             seq_length, 
                             1))
    
    # predict character
    yhat = model.predict_classes(encoded, verbose=0)
    
    # reverse map integer to character
    for char, index in mapping.items():
        if index == yhat:
            break
    seed_text += char

print(seed_text)
