import csv
import random




class InputData(object):

  def __init__(self, data_dir='./data/', batch_size=64):
    self.train_data = self.__create_batchs(data_dir)
    self.batch_size = batch_size
    self.idx = 0
  
  def next_batch(self):
    if self.idx + self.batch_size > len(self.train_data):
      self.idx = 0
      random.shuffle(self.train_data)
    batch = self.train_data[self.idx:self.idx+self.batch_size]
    actuals = [actual for (train, actual) in batch]
    trains = [train for (train, actual) in batch]
    self.idx += self.batch_size
    return trains, actuals
  
  
  def __create_batchs(self, data_dir):
    osaka_avgs, osaka_maxs, osaka_mins = self.__read_data(data_dir + 'osaka.csv')
    tokyo_avgs, tokyo_maxs, tokyo_mins = self.__read_data(data_dir + 'tokyo.csv')
    train_data = []
    for i in range(len(osaka_avgs) - 8):
      train_data.append(([osaka_avgs[i:i+7], osaka_maxs[i:i+7], osaka_mins[i:i+7]], [tokyo_avgs[i+8], tokyo_maxs[i+8], tokyo_mins[i+8]]))
    return train_data

  
  def __read_data(self, data_path):
    avgs = []
    maxs = []
    mins = []
    for _avg, _max, _min in self.__read_one_data(data_path):
      avgs.append(_avg)
      maxs.append(_max)
      mins.append(_min)
    return avgs, maxs, mins

  def __read_one_data(self, data_path):
    with open(data_path, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        yield float(row[0]), float(row[1]), float(row[2])