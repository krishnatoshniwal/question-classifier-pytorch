import json
import os.path as osp
import numpy
from utils import save_obj


def get_data(file_name, quest_freq):
    all_ques = []
    all_ans = []

    with open(file_name) as json_data:
        data = json.load(json_data)['data']
        for dat in data:
            story = dat['story']
            questions = []

            for quest in dat['questions']:
                quest = quest['input_text'].replace('?', '').strip()
                num_words = len(quest.split(' '))

                if num_words > 3:
                    if numpy.random.uniform() < quest_freq:
                        questions.append(quest + '?')
                    else:
                        questions.append(quest)

            answers = story.replace('\n', '.').replace('\\', '').replace('\'', '').replace(',', '').split('.')

            for answer in answers:
                num_words = len(answer.split(' '))
                if num_words > 5:
                    if '?' not in answer:
                        all_ans.append(answer.strip())

            all_ques.extend(questions)

    data = []
    for question in all_ques:
        data.append((question, 0))

    for answer in all_ans:
        data.append((answer, 1))

    return data


def make_qdb():
    data_dir = "../data/coqa"
    train_file = osp.join(data_dir, 'coqa-train-v1.0.json')
    val_file = osp.join(data_dir, 'coqa-dev-v1.0.json')
    quest_freq = 0.1

    train_data = get_data(train_file, quest_freq)
    val_data = get_data(val_file, quest_freq)

    train_qdb_fn = osp.join(data_dir, 'train_db')
    val_qdb_fn = osp.join(data_dir, 'val_db')

    save_obj(train_qdb_fn, train_data)
    save_obj(val_qdb_fn, val_data)


def preprocess_test_input(line_input):
    line_ = line_input
    start_brack = line_.find('(')
    end_brack = line_.find(')')

    if start_brack != -1 and end_brack != -1:
        line_ = line_[:start_brack] + ' ' + line_[end_brack + 1:]

    replace_symbols = [',', '%', '  ', r'\\']

    for replace_symb in replace_symbols:
        line_ = line_.replace(replace_symb, '')
    return line_


def read_test_data():
    file_loc = "../data/test/test-inputs.txt"
    test_qdb_fn = '../data/test/test_db'

    with open(file_loc, 'r') as f_:
        text = f_.read()
    lines = text.split('\n')
    lines.pop() # pop the last line
    questions = [preprocess_test_input(line_) for line_ in lines]

    save_obj(test_qdb_fn, questions)


if __name__ == '__main__':
    read_test_data()
    make_qdb()
