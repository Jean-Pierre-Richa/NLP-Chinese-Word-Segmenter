import os
import tensorflow as tf
from tqdm import tqdm
import json
import time

os.chdir('../')
cwd = os.getcwd()
out_path = './datasetOutput/'

def getDir(phase):

    DIR = './resources/dataset/%s'%phase
    DATASET_DIR = os.path.join(cwd, DIR)

    return DATASET_DIR

def createFiles(phase):

    DATASET_DIR = getDir(phase)

    file = os.path.join(out_path, '%s_tags.txt'%phase)
    path = os.path.join(cwd, DATASET_DIR, file)

    return path

"""
    Read the 4 utf8 files in the specified phase (training or gold),
    wrap them into one list, create one file of labels (BIES) from the
    4 utf8 files, then create (1) a unique char to id dict and (2) one
    file of ids from the 4 utf8 files
"""
def generateCharsAndLabels(phase):

    DATASET_DIR = getDir(phase)
    file = createFiles(phase)
    chars = []
    tags_file = open(file, 'w')
    for utf8File in os.listdir(DATASET_DIR):
        if utf8File.endswith('.utf8'):
            with open(os.path.join(cwd, DATASET_DIR, utf8File), 'rb') as fb:
                contentslen = fb.read().decode('UTF8')
            with open(os.path.join(cwd, DATASET_DIR, utf8File), 'rb') as f:
                print("tagging %s file"%utf8File.split('/')[-1])
                contents = f.readline().decode('UTF-8')
                with tqdm(desc="chars & ids", total=len(contentslen)) as pbar:
                    while contents:
                        contents = f.readline().decode('UTF-8')
                        for x in contents.split():
                            if(len(x) == 1):
                                tag = "S"
                            elif (len(x) == 2):
                                tag = "BE"
                            elif(len(x) == 3):
                                tag = "BIE"
                            elif(len(x)>3):
                                tag = "B"+str((len(x)-2)*"I")+"E"
                            chars.append(x)
                            tags_file.write(tag)
                        tags_file.write('\n')
                        pbar.update(len(contents))
    # Creates the unique char to id dict from the 4 utf8 files
    unique_char_to_id(phase, chars)
    # Converts the 4 utf8 files contents into 1 file char to ids
    char_to_id(phase, chars)

"""
    Takes the phase (training or gold), a list wrapping the 4 utf8 files as input
    and writes a txt file converted from chars to ids
"""
def char_to_id(phase, chars):

    DATASET_DIR = getDir(phase)
    output_file = os.path.join(DATASET_DIR, out_path, 'char_to_id.txt')
    word_to_id_dict = os.path.join(DATASET_DIR, out_path, 'unique_char_to_id.json')

    char_to_id_file = open(output_file, 'w')
    all_dict = jsonToDict(word_to_id_dict)
    print("Json dict loaded successfully")

    print("Converting the wrapped chars to ids")
    with tqdm(desc="chars->ids", total=len(chars)) as pbar:
        for setChar in chars:
            pbar.update(1)
            if len(setChar)==1:
                for key1, value1 in all_dict.items():
                    if (len(key1) == 1) and (key1 == setChar):
                        char_to_id_file.write(str(value1))
                        char_to_id_file.write(',')
            elif len(setChar)>1:
                for char in setChar:
                    for key2, value2 in all_dict.items():
                        if char == key2:
                            char_to_id_file.write(str(value2))
                            char_to_id_file.write(',')
    char_to_id_file.close()

"""
    Takes as input the path to a json file and loads it into a dict
"""

def jsonToDict(path):

    print("loading json file from %s"%path.split('/')[-1])
    # DATA_PATH = getDir(path)
    json_file = os.path.join(cwd, path)

    with open(json_file) as fDict:
        final_dict = json.load(fDict)

    return final_dict

"""
    Takes as input the phase (training or gold) and a list of characters
    (here the characters wrapped from 4 different utf8 files sent from the
    generateCharsAndLabels function) and creates a json file containing unique
    ids from unigrams and bigrams
"""
def unique_char_to_id(phase, chars):

    DATASET_DIR = getDir(phase)
    output_file = os.path.join(DATASET_DIR, out_path, 'unique_char_to_id.json')
    dictPath = open(output_file, 'w')

    word_to_id_dict = {}
    word_to_id_dict["<PAD>"] = 0
    word_to_id_dict["<START>"] = 1
    word_to_id_dict["<UNK>"] = 2
    bigram_id = 3
    unigram_id = 0

    print("Creating the unique bigram chars to ids dict")
    with tqdm(desc="Unique-bigrams-chars->ids", total=len(chars)) as pbar:
        for x in chars:
            pbar.update(1)
            if (len(x) > 1):
                for i in range(len(x)-1):
                    if x[i:i+2] in word_to_id_dict.keys():
                        pass
                    else:
                        word_to_id_dict[x[i:i+2]]=bigram_id
                        bigram_id+=1
                    i+=1
    unigram_id = bigram_id
    print("found %s unique bigrams in "%bigram_id, phase)
    print("Creating the unique unigram chars to ids dict")
    with tqdm(desc="Unique-unigrams-chars->ids", total=len(chars)) as ppbar:
        for y in chars:
            ppbar.update(1)
            for char in y:
                if char in word_to_id_dict.keys():
                    continue
                else:
                    word_to_id_dict[char]=unigram_id
                    unigram_id+=1
    print("found %s unique unigrams in "%unigram_id, phase)
    json.dump(word_to_id_dict, dictPath)
    dictPath.close()


def main(_):
    generateCharsAndLabels('training')
    generateCharsAndLabels('gold')


if __name__ == '__main__':
  tf.app.run()
