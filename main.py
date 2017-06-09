import tensorflow as tf
import random
import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


TRAIN_SIZE = 7
ACTUAL_SIZE = 3
MAX_STEP = 3000


def train(loss):
  return tf.train.AdamOptimizer().minimize(loss)


def predicrt(x, rate):
  output = tf.contrib.layers.flatten(x)
  output = tf.layers.dense(output, 128)
  output = tf.nn.relu(output)
  output = tf.layers.dropout(output, rate=rate)
  output = tf.layers.dense(output, 32)
  output = tf.nn.relu(output)
  output = tf.layers.dropout(output, rate=rate)
  output = tf.layers.dense(output, 16)
  output = tf.nn.relu(output)
  output = tf.layers.dropout(output, rate=rate)
  output = tf.layers.dense(output, 3)
  return output

def loss(predict, actual):
  return l2loss(tf.reduce_mean(tf.pow(predict - actual, 2)))

def l2loss(loss):
  variables = tf.trainable_variables()
  for v in variables:
    if 'kernel' in v.name:
      loss += tf.nn.l2_loss(v)
  return loss

def generate_graph(actual, predict, name):
  fig = plt.figure()

  ax = fig.add_subplot(1,1,1)

  ax.scatter(actual, predict)

  ax.set_title(name)
  ax.set_xlabel('actual')
  ax.set_ylabel('predict')

  plt.savefig('./output/%s.png' % name)


def main():
  data = input_data.InputData()

  x = tf.placeholder(tf.float32, shape=[None, 3, 9])
  y = tf.placeholder(tf.float32, shape=[None, 3])
  rate = tf.placeholder(tf.float32, shape=[])
  predict_op = predicrt(x, rate)
  loss_op = loss(predict_op, y)
  train_op = train(loss_op)
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    for i in range(MAX_STEP):
      train_data, actual_data = data.next_batch()
      _, _predict, _loss = sess.run([train_op, predict_op, loss_op], feed_dict={x: train_data, y: actual_data, rate: 0.5})
      if i % 100 == 0:
        print('step: %05d loss: %0.05f' % (i, _loss))
      if i % 1000 == 0:
        test_data, test_target = data.test_data()
        _loss, _predict = sess.run([loss_op, predict_op], feed_dict={x: test_data, y: test_target, rate: 1.0})
        sample_predict_index = random.choice(range(len(_predict)))
        sample_predict = _predict[sample_predict_index]
        actual_tmp = test_target[sample_predict_index] 
        print('Test loss: %00.05f sample_avg_tmp: %00.02f sample_max_tmp: %00.02f sample_min_tmp: %00.02f actual_avg_tmp: %00.02f actual_max_tmp: %00.02f actual_min_tmp: %00.02f'
           % (_loss, sample_predict[0], sample_predict[1], sample_predict[2], actual_tmp[0], actual_tmp[1], actual_tmp[2]))
    
    _predicts = sess.run(predict_op, feed_dict={x: test_data, y: test_target, rate: 1.0})
    actual_avg_tmps = [d[0] for d in test_target]
    actual_max_tmps = [d[1] for d in test_target]
    actual_min_tmps = [d[2] for d in test_target]
    predict_avg_tmps = [p[0] for p in _predicts]
    predict_max_tmps = [p[1] for p in _predicts]
    predict_min_tmps = [p[2] for p in _predicts]
    generate_graph(actual_avg_tmps, predict_avg_tmps, 'average')
    generate_graph(actual_max_tmps, predict_max_tmps, 'max')
    generate_graph(actual_min_tmps, predict_min_tmps, 'min')
    # _predict = sess.run(predict_op, feed_dict={x:[[[20.5, 19.4, 20.3, 20.5, 21.4, 19.2, 21.2, 20.3, 22.0], [25.9, 24.5, 26.1, 27.0, 26.4, 22.6, 26.3, 22.6, 25.8], [16.9, 15.5, 15.7, 15.1, 16.2, 17.6, 17.9, 17.9, 19.2]]], rate: 1.0})
    # print(_predict)

if __name__ == '__main__':
  main()

