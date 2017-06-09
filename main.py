import tensorflow as tf

import input_data


TRAIN_SIZE = 7
ACTUAL_SIZE = 3
MAX_STEP = 10000


def train(loss):
  return tf.train.AdamOptimizer().minimize(loss)

def predicrt(x):
  print(x)
  output = tf.contrib.layers.flatten(x)
  output = tf.layers.dense(output, 50)
  output = tf.nn.relu(output)
  output = tf.layers.dense(output, 10)
  output = tf.nn.relu(output)
  output = tf.layers.dense(output, 3)
  return output

def loss(predict, actual):
  return tf.reduce_mean(tf.pow(predict - actual, 2))

def main():
  data = input_data.InputData()

  x = tf.placeholder(tf.float32, shape=[None, 3, 7])
  y = tf.placeholder(tf.float32, shape=[None, 3])
  predict_op = predicrt(x)
  loss_op = loss(predict_op, y)
  train_op = train(loss_op)
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    for i in range(MAX_STEP):
      train_data, actual_data = data.next_batch()
      _, _predict = sess.run([train_op, predict_op], feed_dict={x: train_data, y: actual_data})
      #print(_predict[0], actual_data[0])

if __name__ == '__main__':
  main()

