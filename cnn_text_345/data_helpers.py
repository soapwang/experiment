import numpy as np
import re
import itertools
import codecs
RT_POS = "./data/rt/pos.txt"
RT_NEG = "./data/rt/neg.txt"
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"can\'t", "cannot", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)#替换连续出现两次及以上的空格变成1个空格
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
    negative_examples = list(open(negative_data_file, "r",encoding='utf-8').readlines())
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



def load_data_and_labels_fine_grained(star1, star2, star3, star4, star5):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # # Load data from files
    # positive_examples = list(open(positive_data_file, "r",encoding='utf-8').readlines())
    # # print("11111",type(positive_examples))
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r",encoding='utf-8').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # # print(type(positive_examples))
    # x_text = positive_examples[0:25000] + negative_examples[0:25000]
    # x_text = [clean_str(sent) for sent in x_text]
    #
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples[0:25000]]
    # negative_labels = [[1, 0] for _ in negative_examples[0:25000]]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    # return [x_text, y]

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


# load pre-trained word vectors
def load_word_vectors(filename):
    words_list = []
    word_vectors = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip('\n')
            split = line.split(" ")
            words_list.append(split[0])
            word_vectors.append(split[1:])
        word_vectors = np.array(word_vectors)
        return words_list, word_vectors
        
        
# only load the most frequent 100K words        
def load_word_vectors_small(filename):
    words_list = []
    word_vectors = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        lines = f.readlines()
        lines = lines[:100000]
        for line in lines:
            line = line.strip('\n')
            split = line.split(" ")
            words_list.append(split[0])
            word_vectors.append(split[1:])
        word_vectors = np.array(word_vectors)
        return words_list, word_vectors


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


# shape of id matrix = lines*max_seq_length
def load_movie_reviews(words_list,  max_seq_length):
    labels = []
    with codecs.open(RT_POS, "r", "utf-8") as f:
        line_counter = 0
        lines=f.readlines()
        ids_pos = np.zeros((len(lines), max_seq_length), dtype='int32')
        for line in lines:
            index_counter = 0
            cleaned = clean_str(line)
            split = cleaned.split()
            labels.append([1, 0])
            for word in split:
                try:
                    ids_pos[line_counter][index_counter] = words_list.index(word)
                except ValueError:
                    ids_pos[line_counter][index_counter] = 399999 #Vector for unkown words

                index_counter += 1

                if index_counter >= max_seq_length:
                    break
            line_counter += 1

    with codecs.open(RT_NEG, "r", "utf-8") as f:
        line_counter = 0
        lines=f.readlines()
        ids_neg = np.zeros((len(lines), max_seq_length), dtype='int32')
        for line in lines:
            index_counter = 0
            cleaned = clean_str(line)
            split = cleaned.split()
            labels.append([0, 1])
            for word in split:
                try:
                    ids_neg[line_counter][index_counter] = words_list.index(word)
                except ValueError:
                    ids_neg[line_counter][index_counter] = 399999 #Vector for unkown words

                index_counter += 1

                if index_counter >= max_seq_length:
                    break
            line_counter += 1


    np.save('idsMatrix1', ids_pos) #Save the id matrix to a binary file in NumPy .npy format.
    # print("shape of positive ids ", ids_pos.shape)
    np.save('idsMatrix2', ids_neg)
    labels = np.array(labels)
    np.save('labels', labels)
    # print("shape of negative ids ", ids_neg.shape)
    print("ID matrix saved.")