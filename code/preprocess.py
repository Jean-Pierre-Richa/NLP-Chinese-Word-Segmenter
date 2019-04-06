import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

os.chdir('../')
cwd = os.getcwd()

def getDir(phase):

    DIR = './resources/dataset/' + phase
    TRAIN_DIR = os.path.join(cwd, DIR)

    return TRAIN_DIR

def createFiles(phase, TRAIN_DIR):

    if os.path.isdir('%s/datasetOutput/'%TRAIN_DIR):
        print('Output dir already exists.')
    else:
        print('Creating output dir.')
        os.mkdir('%s/datasetOutput'%TRAIN_DIR)
    out_path = './datasetOutput/'

    trainOutFile = os.path.join(out_path, '_train.txt')
    trainOutPath = open(os.path.join(cwd, TRAIN_DIR, trainOutFile),'w')
    labelOutFile = os.path.join(out_path, '_label.txt')
    labelOutPath = open(os.path.join(cwd, TRAIN_DIR, labelOutFile),'w')

    return trainOutPath, labelOutPath

def generateSet(phase, TRAIN_DIR):

    train, label = createFiles(phase, TRAIN_DIR)
    for utf8File in os.listdir(TRAIN_DIR):
        if utf8File.endswith('.utf8'):
            with open(os.path.join(cwd, TRAIN_DIR, utf8File), 'rb') as f:
                print('reading', f)
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

def generateTrainSet():
    phase = 'training'
    TRAIN_DIR = getDir(phase)
    generateSet(phase, TRAIN_DIR)

def generateGoldSet():
    phase = 'gold'
    TRAIN_DIR = getDir(phase)
    generateSet(phase, TRAIN_DIR)

def main(_):
  generateTrainSet()
  generateGoldSet()

if __name__ == '__main__':
  tf.app.run()
