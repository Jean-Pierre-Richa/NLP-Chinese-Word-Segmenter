from argparse import ArgumentParser
from preprocess import char_to_id
from preprocess import jsonToDict
from tensorflow.keras.models import load_model
from network import create_model
import os
import json
from preprocess import generate
from shutil import copyfile
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from score import score
from network import textToList

os.chdir('./public-homework-1-final/')
cwd = os.getcwd()
# cwd = os.getcwd()
print('cwd ', cwd)
MAX_LENGTH = 1700

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


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

    print("loading json file from %s"%json_file.split('/')[-1])

    with open(json_file) as fDict:
        final_dict = json.load(fDict)

    idKey = list(final_dict.values())
    vocab_size = int(idKey[-1])+1
    print('vocab size', vocab_size)

    return vocab_size

def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    true_tags = os.path.join(resources_path, "dataset_new/predict/true_tags.txt")
    input_path = os.path.join(cwd, resources_path, input_path)
    output_path = os.path.join(cwd, resources_path, output_path)
    true_tags_path = os.path.join(cwd, true_tags)

    src = os.path.join(cwd, "%sdataset_new/all/datasetOutput/unique_unigrams_char_to_id.json"%resources_path)
    dst = os.path.join(cwd, "%sdataset_new/predict/datasetOutput/unique_unigrams_char_to_id.json"%resources_path)
    copyfile(src, dst)

    VOCAB_SIZE = get_vsize('unigrams')
    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 100

    idText = []

    ckpt = os.path.join(cwd, "%sweights.hdf5"%resources_path)
    model = create_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)

    with open(input_path, 'rb') as fb:
        raw_text = fb.read().decode('UTF-8')

    lines = []
    chars_file = []
    # phase = 'training'
    with open(input_path, 'rb') as f:
        content = f.readline().decode('UTF-8')
        while content:
            lines.append(content)
            content = f.readline().decode('UTF-8')

    _, tags = generate(lines, 'predict')

    idText = char_to_id('predict', lines, 'unigrams')
    # print(idText)
    idText = pad_sequences(idText, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    labels_pred = model.predict(np.array(idText))
    label_i = np.argmax(labels_pred, axis=2)

    prediction = []

    labels_dict = {1:'B', 2:'I', 3:'E', 4:'S'}
    # tags2 = []

    with tqdm(desc="prediction file", total=len(idText)) as pbar:
        for i in range(len(label_i)):
            pbar.update(1)
            sentence = label_i[i]
            labels = []
            num_char = np.count_nonzero(idText[i])
            for char in sentence[0:num_char]:
                if (char != 0):
                    labels.append(labels_dict[char])
            prediction.append(labels)

    # Write the predicted output
    with open(output_path, "w") as p_t:
        for p_tag in prediction:
            for e in p_tag:
                p_t.write(''.join(str(e)))
            p_t.write('\n')

    # write the correct tags file
    with open(true_tags_path, "w") as t_t:
        for t_tag in tags:
            for e in t_tag:
                t_t.write(''.join(str(e)))
            t_t.write('\n')

    pred_list = textToList(true_tags_path, 'labels')
    true_list = textToList(output_path, 'labels')

    for i in range(len(pred_list)):
        if (len(pred_list[i]) != len(true_list[i])):
            diff = len(pred_list[i])-len(true_list[i])
            del pred_list[i][-diff:]
    score(pred_list, true_list, verbose=True)

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
