import numpy as np
import re
import itertools
from collections import Counter
import codecs


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9\'\`]", " ", string)
    string = re.sub(r"can\'t", "cannot", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string) #替换连续出现两次及以上的空格变成1个空格
    return string.strip().lower()


def load_data_and_labels(positive_data_file,negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    positive_examples = list(open(positive_data_file, "r",encoding='utf-8').readlines())
    # print("11111",type(positive_examples))
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(codecs.open(negative_data_file, "r",encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    # print(type(positive_examples))
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_yelp_polarity(yelp_file, small_data):
    import codecs
    import json
    x_text = []
    y = []
    i = 0
    j = 0
    with codecs.open(yelp_file, "r", "utf-8") as f:
        for line in f:
            j += 1
            review = json.loads(line)
            stars = review["stars"]
            text = review["text"]
            if stars != 3:
                text = clean_str(text.strip())
                if stars == 1 or stars == 2:
                    y.append([1, 0])
                    x_text.append(text)
                elif stars == 4 or stars == 5:
                    y.append([0, 1])
                    x_text.append(text)
                i += 1
                if i % 10000 == 0:
                    print("Non-neutral instances processed: " + str(i))
            if small_data:
                if j == 50000:
                    break;
    y = np.array(y)
    return [x_text, y]


def load_yelp_full(yelp_file, small_data):
    import codecs
    import json
    x_text = []
    y = []
    i = 0
    with codecs.open(yelp_file, "r", "utf-8") as f:
        for line in f:
            review = json.loads(line)
            stars = review["stars"]
            text = review["text"]
            text = clean_str(text.strip())
            if stars == 1:
                y.append([1, 0, 0, 0, 0])
                x_text.append(text)
            elif stars == 2:
                y.append([0, 1, 0, 0, 0])
                x_text.append(text)
            elif stars == 3:
                y.append([0, 0, 1, 0, 0])
                x_text.append(text)
            elif stars == 4:
                y.append([0, 0, 0, 1, 0])
                x_text.append(text)
            else:
                y.append([0, 0, 0, 0, 1])
                x_text.append(text)
            i += 1
            if i % 10000 == 0:
                print("Non-neutral instances processed: " + str(i))
            if small_data:
                if i == 50000:
                    break;
    y = np.array(y)
    return [x_text, y]


def load_imdb_train(imdb_files):

    for f in imdb_files:
        train_pos = list(open(imdb_files[0], "r").readlines())
        train_pos = [s.strip() for s in train_pos]
        train_neg = list(open(imdb_files[1], "r").readlines())
        train_neg = [s.strip() for s in train_neg]
        x_text = train_pos + train_neg
        x_text = [clean_str(sent) for sent in x_text]

        pos_labels = [[1, 0] for _ in train_pos]
        neg_labels = [[0, 1] for _ in train_neg]

        y = np.concatenate([pos_labels, neg_labels], 0)
    return [x_text, y]


def load_imdb_test(imdb_files):
    for f in imdb_files:
        test_pos = list(open(imdb_files[0], "r").readlines())
        test_pos = [s.strip() for s in test_pos]
        test_neg = list(open(imdb_files[1], "r").readlines())
        test_neg = [s.strip() for s in test_neg]
        x_text = test_pos + test_neg
        x_text = [clean_str(sent) for sent in x_text]

        pos_labels = [[1, 0] for _ in test_pos]
        neg_labels = [[0, 1] for _ in test_neg]

        y = np.concatenate([pos_labels, neg_labels], 0)
    return [x_text, y]

def load_data_and_labels_fine_grained(star1, star2, star3, star4, star5):

    star_1 = list(open(star1, "r",encoding='utf-8').readlines())
    # print("11111",type(positive_examples))
    star_1 = [s.strip() for s in star_1]
    star_2 = list(open(star2, "r",encoding='utf-8').readlines())
    star_2 = [s.strip() for s in star_2]
    star_3 = list(open(star3, "r",encoding='utf-8').readlines())
    star_3 = [s.strip() for s in star_3]
    star_4 = list(open(star4, "r",encoding='utf-8').readlines())
    star_4 = [s.strip() for s in star_4]
    star_5 = list(open(star5, "r",encoding='utf-8').readlines())
    star_5 = [s.strip() for s in star_5]
    # Split by words
    # print(type(positive_examples))
    x_text = star_1+star_2+star_3+star_4+star_5
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    star_1_labels = [[1, 0, 0, 0, 0] for _ in star_1]
    star_2_labels = [[0, 1, 0, 0, 0] for _ in star_2]
    star_3_labels = [[0, 0, 1, 0, 0] for _ in star_3]
    star_4_labels = [[0, 0, 0, 1, 0] for _ in star_4]
    star_5_labels = [[0, 0, 0, 0, 1] for _ in star_5]


    y = np.concatenate([star_1_labels, star_2_labels,star_3_labels,star_4_labels,star_5_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
