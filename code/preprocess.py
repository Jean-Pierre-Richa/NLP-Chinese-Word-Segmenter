import os
import tensorflow as tf
from tqdm import tqdm
import json
import time

os.chdir('../')
cwd = os.getcwd()
out_path = './datasetOutput/'

def getDir(phase):

    DIR = './resources/dataset_new/%s'%phase
    DATASET_DIR = os.path.join(cwd, DIR)

    return DATASET_DIR

def createFolders(phase):

    TRAIN_DIR = getDir(phase)

    if not os.path.isdir('%s/datasetOutput/'%TRAIN_DIR):
        os.mkdir('%s/datasetOutput'%TRAIN_DIR)
        print('Creating %s/datasetOutput folder.'%phase)
    else:
        pass

def createTagFile(phase):

    createFolders(phase)
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
def generateCharsAndLabels(phase, uni_or_bi):

    DATASET_DIR = getDir(phase)

    lines = []

    for utf8File in os.listdir(DATASET_DIR):
        if utf8File.endswith('.utf8'):
            file_to_convert = os.path.join(cwd, DATASET_DIR, utf8File)
            print("tagging %s file"%utf8File.split('/')[-1])
            with open(file_to_convert, 'rb') as f:
                contents = f.readline().decode('UTF-8')
                while contents:
                    contents = f.readline().decode('UTF-8')
                    lines.append(contents)


    chars, _ = generate(lines, phase)
    # Creates the unique char to id dict from the 4 utf8 files
    unique_char_to_id(phase, chars, uni_or_bi)
    # Converts the 4 utf8 files contents into 1 file char to ids
    _ = char_to_id(phase, lines, uni_or_bi)

def generate(lines, phase):
    chars = []
    tags = []
    file = createTagFile(phase)
    tags_file = open(file, 'w')
    with tqdm(desc="chars & ids", total=len(lines)) as pbar:
        for line in lines:
            chars_line = []
            tags_line = []
            pbar.update(1)
            for x in line.split():
                if(len(x) == 1):
                    tag = "S"
                elif (len(x) == 2):
                    tag = "BE"
                elif(len(x) == 3):
                    tag = "BIE"
                elif(len(x)>3):
                    tag = "B"+str((len(x)-2)*"I")+"E"
                tags_file.write(tag)
                if phase != 'predict':
                    chars.append(x)
                if phase == 'predict':
                    tags_line.append(tag)
                    chars_line.append(x)
            if phase == 'predict':
                chars.append(chars_line)
                tags.append(tags_line)
            tags_file.write('\n')
    return chars, tags


"""
    Takes the phase (training or gold), a list wrapping the 4 utf8 files as input
    and writes a txt file converted from chars to ids
"""
def char_to_id(phase, lines, uni_or_bi):

    DATASET_DIR = getDir(phase)
    chars_num = 0
    if (uni_or_bi == 'unigrams'):
        word_to_id_dict = os.path.join(DATASET_DIR, out_path, 'unique_unigrams_char_to_id.json')
        output_file = os.path.join(DATASET_DIR, out_path, 'uni_char_to_id.txt')

    char_to_id_file = open(output_file, 'w')
    all_dict = jsonToDict(word_to_id_dict)
    print("Json dict loaded successfully")

    print("Converting the wrapped chars to ids")

    sentence_toId = []

    with tqdm(desc="chars & ids", total=len(lines)) as pbar:
        if uni_or_bi == 'unigrams':
            for x in lines:
                char_toId = []
                pbar.update(1)
                for char in x:
                    if char in all_dict.keys():
                        char_to_id_file.write(str(all_dict[char]))
                        char_to_id_file.write(',')
                        char_toId.append(all_dict[char])
                    # else:
                        # char_to_id_file.write(str(all_dict['<UNK>']))
                        # char_to_id_file.write(',')
                        # char_toId.append(all_dict['<UNK>'])
                        # print("%s is <UNK>"%char)
                sentence_toId.append(char_toId)
                char_to_id_file.write('\n')
            # print(sentence_toId)
    return sentence_toId

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
def unique_char_to_id(phase, chars, uni_or_bi):

    DATASET_DIR = getDir(phase)
    uni_unique_file = os.path.join(DATASET_DIR, out_path, 'unique_unigrams_char_to_id.json')
    bi_unique_file = os.path.join(DATASET_DIR, out_path, 'unique_bigrams_char_to_id.json')

    if (uni_or_bi == 'unigrams'):
        dict_path = open(uni_unique_file, 'w')

    word_to_id_dict = {}
    wordlist = []

    word_to_id_dict["<PAD>"] = 0
    word_to_id_dict["<UNK>"] = 1

    if(uni_or_bi == 'unigrams'):
        id = 2
        print("Creating the unique unigram chars to ids dict")
        with tqdm(desc="Unique-unigrams-chars->ids", total=len(chars)) as ppbar:
            for y in chars:
                ppbar.update(1)
                if len(y)>0:
                    for char in y:
                        if char in word_to_id_dict.keys():
                            continue
                        else:
                            wordlist.append(char)
                            if(wordlist.count(char)) >= 10:
                                word_to_id_dict[char]=id
                                id+=1
                            else:
                                pass
                else:
                    pass
        print("found %s unique unigrams in "%id, phase)
        json.dump(word_to_id_dict, dict_path)
        dict_path.close()
def main(_):
    generateCharsAndLabels('all', 'unigrams')

if __name__ == '__main__':
  tf.app.run()
