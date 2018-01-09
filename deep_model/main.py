import numpy as np
import time
from deep_model.model import model
from deep_model.predict import predict
from six.moves import cPickle as pickle

start_time=time.clock()

pickle_file = 'C:/Users/deep1/PycharmProjects/Deep_Learning_Udacity/data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
train_dataset=train_dataset.T
valid_dataset=valid_dataset.T
test_dataset=test_dataset.T
train_labels=train_labels.T
valid_labels=valid_labels.T
test_labels=test_labels.T
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


parameters = model(train_dataset, train_labels, valid_dataset, valid_labels, learning_rate=0.0005,lambd=0.1,num_epochs=10, minibatch_size=128, print_cost=True)

predict(test_dataset, test_labels, parameters)
end_time=time.clock()
print(end_time-start_time)