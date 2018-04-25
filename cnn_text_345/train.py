#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
# from text_cnn1 import TextCNN1
from text_cnn_rnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("happy", "./data/weibo_data/0happy.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("anger", "./data/weibo_data/1anger.txt", "Data source for the negative data.")
# tf.flags.DEFINE_string("hate", "./data/weibo_data/2hate.txt", "Data source for the negative data.")
# tf.flags.DEFINE_string("low", "./data/weibo_data/3low.txt", "Data source for the negative data.")

#
# tf.flags.DEFINE_string("star1", "./data/yelp_fine_grained/1_star.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("star2", "./data/yelp_fine_grained/2_star.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("star3", "./data/yelp_fine_grained/3_star.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("star4", "./data/yelp_fine_grained/4_star.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("star5", "./data/yelp_fine_grained/5_star.txt", "Data source for the positive data.")


tf.flags.DEFINE_string("positive_data_file", "./data/rt/pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt/neg.txt", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "./data/yelp_pos.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/yelp_neg.txt", "Data source for the negative data.")

tf.flags.DEFINE_string("GLOVE_100", "C:/Users/Soapwang/PycharmProjects/datasets/glove.6B.100d.txt",
                       "Pre-trained Glove 100d vectors.")
tf.flags.DEFINE_string("GLOVE_300", "C:/Users/Soapwang/PycharmProjects/datasets/glove.6B.300d.txt",
                       "Pre-trained Glove 300d vectors.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.happy, FLAGS.anger, FLAGS.hate, FLAGS.low)
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# print("x_text::\n",x_text[0])
# print(x_text[1])
# Build vocabulary from the input data
'''
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("shapes:", x.shape, y.shape) # x:(10662, 56) y:(10662, 2)
'''


# we use pre-defined words list instead
# 78% test acc on MR using GLOVE_100 & GLOVE_300
max_document_length = 56
words_list, word_vectors = data_helpers.load_word_vectors(FLAGS.GLOVE_100)
print("Word vectors loaded...")
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# data_helpers.load_movie_reviews(words_list, max_document_length)
ids_pos = np.load("idsMatrix1.npy")
ids_neg = np.load("idsMatrix2.npy")
y = np.load("labels.npy")

x = np.concatenate([ids_pos, ids_neg], axis=0)
# x = np.load("./data/yelp_polarity/yelp_polarity_500K.npy")
# y = np.load("./data/yelp_polarity/labels_polarity_500K.npy")
print("Loaded id matrices.")
print("shapes:", x.shape, y.shape)


# Randomly shuffle data
np.random.seed(7)
shuffle_indices = np.random.permutation(np.arange(len(y)))
y_shuffled = y[shuffle_indices]
x_shuffled = x[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# 1121 for 50K yelp data
# but the avg length is around 120
# so we have a very large amount of zeroes in training data
print("seq_length=%d" % x_train.shape[1])


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=400000,
            # vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            embedding_mat= word_vectors,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # print(x_batch[0])
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step%100==0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            x_dev = np.array(x_batch)
            dev_size = len(x_batch)
            max_batch_size = 100
            num_batches = dev_size // max_batch_size
            acc = []
            losses = []
            for i in range(num_batches):
                x_batch_dev = x_dev[i * max_batch_size:(i + 1) * max_batch_size]
                y_batch_dev = y_dev[i * max_batch_size:(i + 1) * max_batch_size]
                feed_dict = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                acc.append(accuracy)
                losses.append(loss)

            time_str = datetime.datetime.now().isoformat()
            print("test {}: step {}, loss {:g}, acc {:g}".format(time_str, step, sum(losses)/len(losses),sum(acc)/len(acc)))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
