import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, embedding_mat, l2_reg_lambda=0.0,):
        # rnn参数
        max_time = sequence_length # 56 for MR data
        n_input = embedding_size # default:100
        n_hidden = 256  # 每个rnn cell中隐状态的数量

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")



        # self.embedding_matrix = tf.placeholder(tf.float32, [vocab_size, embedding_size], name="embedding_matrix")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.1)

        # Initialize the embedding matrix with the ndarray
        init = tf.constant(embedding_mat, dtype=tf.float32)
        # init2 = tf.convert_to_tensor(embedding_mat)
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #     name="W", trainable=False)
            self.W = tf.Variable(initial_value=init, name="W", trainable=False)

            # shape = (?, sequence_length, embedding_size), dtype = float32
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # shape = (?, 56, 100, 1), dtype = float32
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # RNN layer

        with tf.name_scope("LSTM_sequence_encoder"):
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

            # time_major = False , outputs中是[batch,max_time,n_hidden]
            # final_state[1]表示最后一个cell的输出。outputs[0]表示第一个数据所有cell输出组合成的矩阵（time_major=False）
            outputs, final_state = tf.nn.dynamic_rnn(cell, self.embedded_chars, dtype=tf.float32, time_major=False)
            sequence_features = final_state[1]


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # 输入长度是56，窄卷积，shape:
                # (?, 54, 1, 256) filter_size = 3
                # (?, 53, 1, 256) filter_size = 4
                # (?, 52, 1, 256) filter_size = 5
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # (?, 1, 1, 256)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_outputs.append(pooled)

        # Combine with rnn features
        sequence_features = tf.expand_dims(sequence_features, 1)
        sequence_features = tf.expand_dims(sequence_features, 1)
        print(sequence_features)  # (?, 1, 1, 256)
        pooled_outputs.append(sequence_features)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + n_hidden
        self.h_pool = tf.concat(pooled_outputs, 3)  # (?, 1, 1, 768)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # (?, 768)


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
