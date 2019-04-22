from argparse import ArgumentParser
from preprocess2 import char_to_id
from preprocess2 import jsonToDict
from tensorflow.keras.models import load_model
from network import create_model
import os
import json
from preprocess2 import generate
from network import id_to_label

# os.chdir('../')
cwd = os.getcwd()

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument("input_path", help="The path of the input file")
#     parser.add_argument("output_path", help="The path of the output file")
#     parser.add_argument("resources_path", help="The path of the resources needed to load your model")
#
#     return parser.parse_args()


def getDir(phase):

    DIR = './public_homework_1/resources/dataset_new/%s'%phase
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

    # longest = max(open(text_file), key=len)
    # for x in longest.split(',')[:-1]:
    #     max_length+=1
    # print('max_length ', max_length)

    return vocab_size

def predict():
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

    resources_path = os.path.join(cwd, "public_homework_1/resources/dataset_new/")
    input_path = os.path.join(resources_path, "gold/as_testing_gold.utf8")
    output_path = os.path.join(resources_path, "output_path.txt")

    VOCAB_SIZE = get_vsize('unigrams')
    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 100

    idText = []

    ckpt = os.path.join(cwd, "checkpoints/weights.hdf5")
    model = create_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)

    with open(input_path, 'rb') as fb:
        raw_text = fb.read().decode('UTF-8')

    # print(idText)
    # print('length id text ', len(idText))
    word_to_id_dict = jsonToDict(os.path.join(cwd, "public_homework_1/resources/dataset_new/all/datasetOutput/unique_unigrams_char_to_id.json"))
    id_to_word_dict = {v:k for k,v in word_to_id_dict.items()}

    # print(id_to_word_dict)

    n_chars = len(raw_text)
    print('chars length ', n_chars)
    n_vocab = len(sorted(list(set(raw_text))))

    lines = []
    chars_file = []
    phase = 'training'
    with open(input_path, 'rb') as f:
        content = f.readline().decode('UTF-8')
        while content:
            content = f.readline().decode('UTF-8')
            lines.append(content)
    chars, tags = generate(lines, 'predict')
    idText = char_to_id('predict', lines, 'unigrams')

    print(idText)

    labels_pred = model.predict(idText)

    prediction = []

    arg = np.argmax(labels_pred, axis=2)

    for i in range(len(arg)):
        sentence = arg[i]
        labels = []
        num_char = np.count_nonzero(idText)
        for char in sentence[0:num_char]:
            labels.append(char)
        prediction.append(labels)
    pred = id_to_label(prediction)
    score(pred, tags, verbose=True)

    with open(output_path, "w+") as f:
        for line in pred:
            f.write(''.join(str(e) for e in line))
            f.write('\n')

    pass

if __name__ == '__main__':
    predict()
    # args = parse_args()
    # predict(args.input_path, args.output_path, args.resources_path)
