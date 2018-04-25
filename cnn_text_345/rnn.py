import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

batch_size = 50
max_time = 28
n_input = 28
n_hidden = 100
n_classes = 10
n_batch = mnist.train.num_examples // batch_size
n_test_batch = mnist.test.num_examples //batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder (tf.float32,[None,10])

weights = tf.Variable(tf.truncated_normal([n_hidden,n_classes],stddev=0.1))
biases = tf.constant(0.1,shape=[n_classes])

def rnn(X,weights,biases):
    # x_input = tf.reshape(X,[max_time,-1,n_input]) #True
    # max_time：  cell数量   n_input:输入向量的维度
    x_input = tf.reshape(X,[-1,max_time,n_input]) #False
    # 创建cell，n_hidden隐藏状态数量
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # time_major = False , outputs中是[batch,max_time,n_hidden]
    #final_state[1]表示最后一个cell的输出。outputs[0]表示第一个数据所有cell输出组合成的矩阵（time_major=False）
    outputs,final_state = tf.nn.dynamic_rnn(cell,x_input,dtype=tf.float32,time_major=False)
    print(outputs)
    print("state::",final_state[1])
    results = tf.matmul(final_state[1],weights)+biases
    return results

#计算rnn返回结果
prediction = rnn(x,weights,biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
train = tf.train.AdamOptimizer(0.001).minimize(loss)

correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(21):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y})

        acc_pre = []
        for batch in range(n_test_batch):
            test_x,test_y = mnist.test.next_batch(batch_size)
            acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
            acc_pre.append(acc)
        print ("Iter" + str(i)+" ,Testing Accuracy" +str(np.mean(acc_pre)))

