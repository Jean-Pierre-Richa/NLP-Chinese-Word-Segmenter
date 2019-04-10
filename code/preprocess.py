import os
import tensorflow as tf
from tqdm import tqdm
import json

os.chdir('../')
cwd = os.getcwd()

def getDir(phase):

    DIR = './resources/dataset/' + phase
    TRAIN_DIR = os.path.join(cwd, DIR)

    return TRAIN_DIR

def createFolders(phase):

    TRAIN_DIR = getDir(phase)

    if not os.path.isdir('%s/datasetOutput'%TRAIN_DIR):
        if phase =='training/final':
            os.mkdir(TRAIN_DIR)
            os.mkdir('%s/datasetOutput'%TRAIN_DIR)
            print('Creating %s/datasetOutput folder.'%phase)

    if not os.path.isdir('%s/datasetOutput/'%TRAIN_DIR):
        os.mkdir('%s/datasetOutput'%TRAIN_DIR)
        print('Creating %s/datasetOutput folder.'%phase)
    else:
        pass

def createFiles(phase):

    TRAIN_DIR = getDir(phase)

    out_path = './datasetOutput/'

    trainOutFile = os.path.join(out_path, '_train.txt')
    trainOutPath = os.path.join(cwd, TRAIN_DIR, trainOutFile)
    labelOutFile = os.path.join(out_path, '_label.txt')
    labelOutPath = os.path.join(cwd, TRAIN_DIR, labelOutFile)
    jsonOutFile = os.path.join(out_path, 'word_to_id.json')
    jsonOutPath = os.path.join(cwd, TRAIN_DIR, jsonOutFile)

    final_jsonFile = os.path.join(out_path, 'word_to_id_final.json')
    final_jsonPath = os.path.join(cwd, TRAIN_DIR, final_jsonFile)


    return trainOutPath, labelOutPath, jsonOutPath, final_jsonPath

def generateSet(phase, TRAIN_DIR):

    train, label, _, _ = createFiles(phase)
    
    train = open(train,'w')
    label = open(label, 'w')

    for utf8File in os.listdir(TRAIN_DIR):
        if utf8File.endswith('.utf8'):
            with open(os.path.join(cwd, TRAIN_DIR, utf8File), 'rb') as f:
                # print('reading', f)
                contents = f.readline().decode('UTF-8')
                count=1
                while contents:
                    contents = f.readline().decode('UTF-8')
                    for char in contents.split():
                        lenChar = len(char)
                        if lenChar>1:
                            beginning = char[:1]
                            bTag = 'B'
                            end = char[-1]
                            eTag = 'E'
                            line = beginning + end
                            tag = bTag + eTag
                            if lenChar>2:
                                inside = char[1:-1]
                                iTag = 'I'
                                line = beginning + inside + end
                                tag = bTag + iTag + eTag
                        elif len(char) == 1:
                            single = char
                            line = char
                            tag = 'S'
                        else:
                            pass
                        train.write(line)
                        label.write(tag)
                    train.write('\n')
                    label.write('\n')
                    count+=1
        else:
            continue

def word_to_id(inputFile, outputFile):

    dictPath = open(outputFile, 'w')

    with open(inputFile, 'r') as f:
        data = f.read()
        id=0
        word_to_id_dict = {}
        for word in data:
            if word in word_to_id_dict.keys():
                continue
            else:
                word_to_id_dict[word]=id
                id+=1
        # print(word_to_id_dict)
        json.dump(word_to_id_dict, dictPath)

def word_to_id_final(json1, json2, final_json):

    final_dict = open(final_json, 'w')

    with open(json1) as json1_file:
        json1_dict = json.load(json1_file)

    with open(json2) as json2_file:
        json2_dict = json.load(json2_file)

    word_to_id_final = {}
    word_to_id_final = json2_dict

    for key, value in word_to_id_final.items():
        if value == len(word_to_id_final)-1:
            last_value = value
            print('Starting from value: ', last_value)

    for key, value in json1_dict.items():
        if key not in word_to_id_final.keys():
            last_value+=1
            word_to_id_final[key]=last_value
    # print(word_to_id_final)
    json.dump(word_to_id_final, final_dict)

def generate_Set(phase, json):
    TRAIN_DIR = getDir(phase)
    generateSet(phase, TRAIN_DIR)
    train_input, label_input, json, _ = createFiles(phase)
    word_to_id(train_input, json)

def main(_):

    createFolders('training')
    createFolders('gold')
    createFolders('training/final')

    generatedSets=0

    _, _, json1, _ = createFiles('training')
    _, _, json2, _ = createFiles('gold')

    generate_Set('training', json1)
    generatedSets+=1
    generate_Set('gold', json2)
    generatedSets+=1
    if(generatedSets == 2):
        _, _, final_json, _ = createFiles('training/final')
        word_to_id_final(json1, json2, final_json)


if __name__ == '__main__':
  tf.app.run()
