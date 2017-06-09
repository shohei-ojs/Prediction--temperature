import csv
import random




class InputData(object):

  def __init__(self, data_dir='./data/', batch_size=64):
    self.train_data = self.__create_batchs(data_dir)
    self.__test_data, self.__test_target = self.__create_test_data(data_dir)
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
  
  
  def __create_batchs(self, data_dir, test=False):
    if not test:
      osaka_avgs, osaka_maxs, osaka_mins = self.__read_data(data_dir + 'osaka.csv')
      tokyo_avgs, tokyo_maxs, tokyo_mins = self.__read_data(data_dir + 'tokyo.csv')
    else:
      osaka_avgs, osaka_maxs, osaka_mins = self.__read_data(data_dir + 'test_osaka.csv')
      tokyo_avgs, tokyo_maxs, tokyo_mins = self.__read_data(data_dir + 'test_tokyo.csv')
    data = []
    for i in range(len(osaka_avgs) - 8):
      data.append(([osaka_avgs[i:i+7] + tokyo_avgs[i+6:i+8], osaka_maxs[i:i+7] + tokyo_maxs[i+6:i+8], osaka_mins[i:i+7] + tokyo_mins[i+6:i+8]]
          , [tokyo_avgs[i+8], tokyo_maxs[i+8], tokyo_mins[i+8]]))
    return data
  
  
  def __create_test_data(self, data_dir):
    test_data = self.__create_batchs(data_dir, test=True)
    actuals = [actual for (train, actual) in test_data]
    _input = [train for (train, actual) in test_data]
    return _input, actuals


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
  
  def test_data(self):
    return self.__test_data, self.__test_target