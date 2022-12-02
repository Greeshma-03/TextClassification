# %%
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model


# %%
############Loading the data###############


train_data_source = 'train.csv'
test_data_source = 'test.csv'

train_df = pd.read_csv(train_data_source, header=None)
test_df = pd.read_csv(test_data_source, header=None)


####### Data Preprocessing part #########

# Each of the row in dataset is (class,title,text)
# combining title and text into second column and dropping that column



train_df[1]=train_df[1]+train_df[2]
test_df[1]=test_df[1]+test_df[2]

train_df=train_df.drop([2],axis=1)
test_df=test_df.drop([2],axis=1)


# convert text_string to lower case 
train_texts = [s.lower() for s in train_df[1].values] 
test_texts = [s.lower() for s in test_df[1].values] 


# %%

# Tokenizer
# Parameteres used:
# num_words=None, since we want to keep all words (here eword referes to character) and not just most frequenet some 'x' words
# oov_token='UNK', as we consider oov words also
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK',lower=True)

# It will understand the train string data that is provided and has direct functions to give results
tk.fit_on_texts(train_texts)


# Vocabulary of size 69 (excluding 'UNK' token)used 
alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" 

#creating word2index dictionary (1-based indexing)
char_dict = {}
for index in range(0,len(alphabet)):
    char_dict[alphabet[index]] = index + 1
    
# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy() 
# Add 'UNK' to the vocabulary 
tk.word_index[tk.oov_token] = len(alphabet) + 1


# Convert the concatenated string to vector of numbers corresponding to the number for each token in vocabulary
train_sequences = tk.texts_to_sequences(train_texts)
test_texts = tk.texts_to_sequences(test_texts)



# %%
# Padding with zeros till maximum length of 1014 and terminating if number of characters have exceeded 1014
train_data=np.array(np.zeros((len(train_sequences),1014)))
test_data=np.array(np.zeros((len(test_texts),1014)))

pos=0
for seq in train_sequences:
    x=pad_sequences([seq], maxlen=1014, padding='post')
    x=np.array(x)
    train_data[pos]=x
    pos+=1
pos=0
for seq in test_texts:
    x=pad_sequences([seq], maxlen=1014, padding='post')
    x=np.array(x)
    test_data[pos]=x
    pos+=1
# train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
# test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')

# Now we are having each row of the train data as 1014 length (format of numpy array)

# %%
print(train_texts[0])
print(train_sequences[0])

print(train_data[0][:30])
print(train_data[0])

# %%
# In the dataframes for train and test df, first dimension corresponds to the class label
train_classes = train_df[0].values #contains just the values of class labels
test_classes = test_df[0].values

# Converting classes to zero based indexing as we are using to_categorical in next step
train_class_list = [int(x)-1 for x in train_classes]
test_class_list = [int(x)-1 for x in test_classes]

# using to_categorical to have the vectorized form of classes (kind of one-hot encoding for classes)
from keras.utils import to_categorical
train_classes = to_categorical(train_class_list)
test_classes = to_categorical(test_class_list)

print(train_classes[0])

# %%
print(train_class_list[0])
print(train_classes[0])

print(tk.word_index)

# %%
vocab_size = len(tk.word_index)
print(vocab_size)

# %%
# First row is meant for padding zeros 
# Remaining 70 rows corresponds to each of the one hot encoding type of the character

embedding_weights = []
zeros=np.zeros(vocab_size)
embedding_weights.append(zeros)

for char, i in tk.word_index.items():
    val = np.zeros(vocab_size)
    val[i-1] = 1
    embedding_weights.append(val)

embedding_weights = np.array(embedding_weights)

# %%
# Here first row is for zero vector and all others correspond to the 70 characters that are considered
print(embedding_weights.shape)
embedding_weights

# %%
########### Model Parameters ################

# input size for each of the sample is 1014 numpy array
input_size = 1014
embedding_size = 70

# For this corpus it is 4
num_of_classes = 4
dropout_p = 0.5

# there are 7 convolution layers as mentioned in research paper
# (number of filters, size of the filter, pooling size)
conv_layers = [[256, 7, 3], 
               [256, 7, 3], 
               [256, 3, 0], 
               [256, 3, 0], 
               [256, 3, 0], 
               [256, 3, 3]]

# Fully connected neural network added to the convolution network
fully_connected_layers = [1024, 256, 64]




# %%
# Model Architecture
from keras.models import Sequential

# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64')

model=Sequential()
# Embedding
model.add(Embedding(vocab_size+1,
              embedding_size,
              input_length=input_size,
              weights=[embedding_weights]))
# Conv
for filter_num, filter_size, pooling_size in conv_layers:
    model.add(Conv1D(filter_num, filter_size))
    model.add(Activation('relu'))
    if pooling_size != 0:
        model.add(MaxPooling1D(pool_size=pooling_size))
model.add(Flatten())

# Adding the fully connected layers
for dense_layer_size in fully_connected_layers:
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout_p))

# Adding a dense output layer equivalent to number of classes and applying softmax activation
model.add(Dense(num_of_classes, activation='softmax'))

# Build model
# Using adam optimizer and cross entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
              'accuracy', 'mse'])  # Adam, categorical_crossentropy

# To print the paremeteres and layers used in the model
model.summary()


# %%
# 1000 training samples and 100 testing samples
indices = []
for i in range(train_classes.shape[0]):
    indices.append(i)
indices=np.array(indices)
np.random.shuffle(indices)

x_train = train_data[indices][:500]
y_train = train_classes[indices][:500]

x_test = test_data[:40]
y_test = test_classes[:40]

# %%
# Training Phase and testing on the created (x_test,y_test)

history=model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=64,
          epochs=5,
          verbose=2)
# %%

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()          
# %%
# matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

predictions=model.predict(x_test)
print(y_test[0])
print(predictions[0])
print(type(predictions[0]))
cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

# sns.heatmap(cf_matrix, annot=True)

cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))

print(cmn)

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=True)



