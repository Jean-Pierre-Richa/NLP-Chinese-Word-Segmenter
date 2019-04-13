import os
import tensorflow as tf
from tqdm import tqdm
import json
import time

os.chdir('../')
cwd = os.getcwd()
out_path = './datasetOutput/'
created = False

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

def createFiles(phase):

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

    global created

    # if not created:
    DATASET_DIR = getDir(phase)
    file = createFiles(phase)
    chars = []

    for utf8File in os.listdir(DATASET_DIR):
        if utf8File.endswith('.utf8'):
            file_to_convert = os.path.join(cwd, DATASET_DIR, utf8File)
            with open(file_to_convert, 'rb') as fb:
                contentslen = fb.read().decode('UTF8')
            with open(file_to_convert, 'rb') as f:
                contents = f.readline().decode('UTF-8')
                if created:
                    while contents:
                        contents = f.readline().decode('UTF-8')
                        for x in contents.split():
                            chars.append(x)
                elif not created:
                    tags_file = open(file, 'w')
                    print("tagging %s file"%utf8File.split('/')[-1])
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
    created = True
    # Creates the unique char to id dict from the 4 utf8 files
    unique_char_to_id(phase, chars, uni_or_bi)
    # Converts the 4 utf8 files contents into 1 file char to ids
    char_to_id(phase, file_to_convert, uni_or_bi)

"""
    Takes the phase (training or gold), a list wrapping the 4 utf8 files as input
    and writes a txt file converted from chars to ids
"""
def char_to_id(phase, file_to_convert, uni_or_bi):

    DATASET_DIR = getDir(phase)

    if (uni_or_bi == 'unigrams'):
        word_to_id_dict = os.path.join(DATASET_DIR, out_path, 'unique_unigrams_char_to_id.json')
        output_file = os.path.join(DATASET_DIR, out_path, 'uni_char_to_id.txt')
    elif (uni_or_bi == 'bigrams'):
        word_to_id_dict = os.path.join(DATASET_DIR, out_path, 'unique_bigrams_char_to_id.json')
        output_file = os.path.join(DATASET_DIR, out_path, 'bi_char_to_id.txt')

    char_to_id_file = open(output_file, 'w')
    all_dict = jsonToDict(word_to_id_dict)
    print("Json dict loaded successfully")

    print("Converting the wrapped chars to ids")
    # with tqdm(desc="chars->ids", total=len(chars)) as pbar:
    with open(file_to_convert, 'rb') as fb:
        contentslen = fb.read().decode('UTF8')
    with open(file_to_convert, 'rb') as f:
        contents = f.readline().decode('UTF-8')
        with tqdm(desc="chars & ids", total=len(contentslen)) as pbar:
            while contents:
                contents = f.readline().decode('UTF-8')
                if uni_or_bi == 'unigrams':
                    for x in contents.split():
                        for char in x:
                            if char in all_dict.keys():
                                char_to_id_file.write(str(all_dict[char]))
                                char_to_id_file.write(',')
                            else:
                                print("%s does not exist in the dictionary"%char)
                    char_to_id_file.write('\n')
                    pbar.update(len(contents))
                elif uni_or_bi == 'bigrams':
                    for x in contents.split():
                        if (len(x) > 1):
                            for i in range(len(x)-1):
                                if x[i:i+2] in all_dict.keys():
                                    char_to_id_file.write(str(all_dict[x[i:i+2]]))
                                    char_to_id_file.write(',')
                                else:
                                    print('%s not found in the dictionary'%x)
                    char_to_id_file.write('\n')
                    pbar.update(len(contents))
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
def unique_char_to_id(phase, chars, uni_or_bi):

    DATASET_DIR = getDir(phase)
    uni_unique_file = os.path.join(DATASET_DIR, out_path, 'unique_unigrams_char_to_id.json')
    bi_unique_file = os.path.join(DATASET_DIR, out_path, 'unique_bigrams_char_to_id.json')

    if (uni_or_bi == 'unigrams'):
        dict_path = open(uni_unique_file, 'w')
    elif (uni_or_bi == 'bigrams'):
        dict_path = open(bi_unique_file, 'w')


    word_to_id_dict = {}

    if(uni_or_bi == 'unigrams'):
        id = 3
        print("Creating the unique unigram chars to ids dict")
        word_to_id_dict["<PAD>"] = 0
        word_to_id_dict["<START>"] = 1
        word_to_id_dict["<UNK>"] = 2
        with tqdm(desc="Unique-unigrams-chars->ids", total=len(chars)) as ppbar:
            for y in chars:
                ppbar.update(1)
                for char in y:
                    if char in word_to_id_dict.keys():
                        pass
                    else:
                        word_to_id_dict[char]=id
                        id+=1
        print("found %s unique unigrams in "%id, phase)
        json.dump(word_to_id_dict, dict_path)
        dict_path.close()
    elif (uni_or_bi == 'bigrams'):

        final_dict = jsonToDict(uni_unique_file)
        idKey = list(final_dict.values())
        id = int(idKey[-1])

        # list(d.items())
        # list[e.keys()[0]]

        print("Creating the unique bigram chars to ids dict for %s"%phase)
        with tqdm(desc="Unique-bigrams-chars->ids", total=len(chars)) as pbar:
            for x in chars:
                pbar.update(1)
                if (len(x) > 1):
                    for i in range(len(x)-1):
                        if x[i:i+2] in word_to_id_dict.keys():
                            pass
                        else:
                            id+=1
                            word_to_id_dict[x[i:i+2]]=id
                        i+=1
        json.dump(word_to_id_dict, dict_path)
        dict_path.close()
        print("found %s unique bigrams in "%id, phase)

    else:
        print('please specify unigrams or bigrams')

def main(_):
    generateCharsAndLabels('all', 'unigrams')
    generateCharsAndLabels('all', 'bigrams')
    # generateCharsAndLabels('gold', 'unigrams')
    # generateCharsAndLabels('training', 'bigrams')
    # generateCharsAndLabels('gold', 'bigrams')


if __name__ == '__main__':
  tf.app.run()
