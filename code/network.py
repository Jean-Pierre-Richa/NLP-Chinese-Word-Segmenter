import os
import numpy as np
import json
import tensorflow as tf
from tqdm import tqdm

os.chdir('../')
cwd = os.getcwd()

def getDir(init_path):

    DIR = './resources/dataset/' + init_path
    DATA_PATH = os.path.join(cwd, DIR)

    return DATA_PATH

def jsonToDict(path):

    print("loading json file from %s"%path.split('/')[-1])
    # DATA_PATH = getDir(path)
    json_file = os.path.join(cwd, path)

    with open(json_file) as fDict:
        final_dict = json.load(fDict)

    return final_dict

def textToList(file):

    print("converting %s file to list"%file.split('/')[-1])
    with open(file, 'r') as f:
        list = f.read().split(',')
    f.close()

    return list

def loadData(data):

    print("loading %s dataset"%data.split('/')[-1])
    # Load data
    list = textToList(data)
    # print('list ', list[:1])

    return list

def charToId(char_list, labels_dict, set):

    print("converting %s words to ids"%set.split('/')[-1])
    charnum = 0
    idnum = 0
    nbhits = 0
    id_list = []
    path = getDir(set)

    id_path = '%s/datasetOutput/_train_id.txt'%path

    if not os.path.isfile(id_path):

        id_file = open(id_path,'w')

        with tqdm(desc="chars->ids", total=len(char_list)) as pbar:
            for x in char_list:
                for char in x:
                    charnum+=1
                    for key, value in labels_dict.items():
                        if (char==key):
                            idnum+=1
                            id_file.write(str(value))
                            id_file.write(',')
                            id_list.append(value)
                            continue
                        else:
                            pass

                pbar.update(1)
        print('charnum ', charnum)
        print('idnum ', idnum)
        pbar.close()
        id_file.close()
    else:
        print("%s file already exists"%set)
        print("loading the file ")
        id_list = loadData(id_path)

    return id_list

def main(_):

    train = getDir('training/datasetOutput/_train.txt')
    dev = getDir('gold/datasetOutput/_train.txt')
    labels = getDir('training/final/datasetOutput/word_to_id.json')
    train_list = loadData(train)
    dev_list = loadData(dev)
    train_labels_dict = jsonToDict(labels)
    train_id_list = charToId(train_list, train_labels_dict, 'training')
    dev_id_list = charToId(dev_list, train_labels_dict, 'gold')


if __name__ == '__main__':
  tf.app.run()
