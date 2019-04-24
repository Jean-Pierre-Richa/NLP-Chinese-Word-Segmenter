import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from time import time


MAX_LENGTH = 300
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 100

os.chdir('../')
cwd = os.getcwd()
ckpt = os.path.join(cwd , "resources/weights.hdf5")

def getDir(phase):

    DIR = './resources/dataset_new/%s'%phase
    DATASET_DIR = os.path.join(cwd, DIR)

    return DATASET_DIR

def get_vsize(uni_or_bi):

    # max_length = 0
    path = getDir('all/datasetOutput/')

    if (uni_or_bi == 'unigrams'):
        json_file = os.path.join(path, 'unique_unigrams_char_to_id.json')
        text_file = os.path.join(path, 'uni_char_to_id.txt')
    elif(uni_or_bi == 'bigrams'):
        json_file = os.path.join(path, 'unique_bigrams_char_to_id.json')
        text_file = os.path.join(path, 'bi_char_to_id.txt')
    else:
        print('please specify unigrams or bigrams')

    print("loading json file from %s"%json_file.split('/')[-1])

    with open(json_file) as fDict:
        final_dict = json.load(fDict)

    idKey = list(final_dict.values())
    vocab_size = int(idKey[-1])+1
    print('vocab size', vocab_size)

    return vocab_size


def textToList(file, set):
    list = []
    print("converting %s file to list"%file.split('/')[-1])
    with open(file, 'r') as f:
        for line in f:
            # if len(line)>1:
            sub_list = []
            if (set == 'chars'):
                for x in line.split(','):
                    if (x != '\n'):
                        sub_list.append(x)
            if (set == 'labels'):
                for x in line:
                    if (x != '\n'):
                        sub_list.append(x)
            list.append(sub_list)
    f.close()

    return list

def batch_generator(data, labels, batch_size):
    batch_data = np.zeros((batch_size, MAX_LENGTH))
    batch_labels = np.zeros((batch_size, MAX_LENGTH, 5))
    while True:
        for i in range(batch_size):
            index = random.randint(0,len(data)-1)
            batch_data[i] = data[index]
            batch_labels[i] = labels[index]
        yield batch_data, batch_labels

def label_to_id(data_list):

    labels_dict = {'B':1, 'I':2, 'E':3, 'S':4}

    data_labeled = []

    for data in data_list:
        data_sub_list = []
        for t_l in data:
            data_sub_list.append(labels_dict[t_l])
        data_labeled.append(data_sub_list)
    return data_labeled


def create_dataset(vocab_size, max_length):

    dataX = getDir('all/datasetOutput/uni_char_to_id.txt')
    dataY = getDir('all/datasetOutput/all_tags.txt')

    print("loading the dataset...")

    data_x = textToList(dataX, 'chars')
    data_y = textToList(dataY, 'labels')

    print('data_x ', len(data_x))
    print('data_y ', len(data_y))

    print("%s dataset successfully loaded"%dataX.split('/')[-1])

    data_y_labeled = label_to_id(data_y)

    data_x = pad_sequences(data_x, truncating='pre', padding='post', maxlen=max_length)
    data_y_labeled = pad_sequences(data_y_labeled, truncating='pre', padding='post', maxlen=max_length, value=0)

    data_y_labeled = to_categorical(data_y_labeled)

    print('length', len(data_x))

    train_x, dev_x, train_y, dev_y = train_test_split(data_x, data_y_labeled, test_size=800)

    print("\nTraining_x set shape:", train_x.shape)
    print("Dev_x set shape:", dev_x.shape)

    return train_x, train_y, dev_x, dev_y

def create_model(vocab_size, embedding_size, hidden_size):


    model = K.models.Sequential()
    model.add(K.layers.Embedding(vocab_size, embedding_size, mask_zero=True))

    model.add(K.layers.Bidirectional(K.layers.LSTM(hidden_size, dropout=0.2,
           recurrent_dropout=0.2, return_sequences=True), merge_mode='concat'))
    model.add(K.layers.Bidirectional(K.layers.LSTM(hidden_size, dropout=0.2,
           recurrent_dropout=0.2, return_sequences=True), merge_mode='concat'))
    model.add(K.layers.Dense(5, activation='softmax'))
    optimizer = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if os.path.isfile(ckpt):
        print("Loading previous checkpoint")
        print('model created')
        model.load_weights(ckpt)
    else:
        print("model created")

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def run_training():


    VOCAB_SIZE = get_vsize('unigrams')
    # MAX_LENGTH = 200
    batch_size = 64
    epochs = 80
    model = create_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)

    model.summary()

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    train_x, train_y, dev_x, dev_y = create_dataset(VOCAB_SIZE, MAX_LENGTH)

    print('train_x', train_x)
    print('train_y', train_y)

    print("\n==============================")
    print("Training size:", len(train_x))
    print("Dev size:", len(dev_x))
    print("==============================\n")

    cbk = K.callbacks.TensorBoard(log_dir="resources/logs/{}".format(time()))
    checkpoint = ModelCheckpoint(ckpt, monitor='acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint, cbk]
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(dev_x, dev_y), callbacks=callbacks_list)
    model.fit_generator(batch_generator(train_x, train_y, batch_size=batch_size),
                                        validation_data=(dev_x, dev_y),
                                        # steps_per_epoch=(len(train_x)/batch_size),
                                        steps_per_epoch=100,
                                        epochs=epochs, callbacks=callbacks_list)
    loss_acc = model.evaluate(dev_x, dev_y, verbose=1)
    model.save_weights(ckpt)
    print("Saved model to disk")

if __name__ == '__main__':
    run_training()
