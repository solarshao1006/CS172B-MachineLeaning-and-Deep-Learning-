import os

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from collections import defaultdict

import warnings


import matplotlib.pyplot as plt
from PIL import Image
torch.cuda.empty_cache()


class load_data:
  def __init__(self, mode:str, train_path="/content/drive/MyDrive/CS172B Final Project/Splited_dataset/train", test_path="/content/drive/MyDrive/CS172B Final Project/Splited_dataset/test", val_path="/content/drive/MyDrive/CS172B Final Project/Splited_dataset/val"):
    """
    Mode: a string specify which data to be loaded. Choice are train, test, val, all
    train_path: a string specify the train data directory path
    test_path: a string specify the test data directory path
    val_path: a string specify the validation data directory path

    Return: 
    A tuple with data dictionary at the first position and count information dictionary at the second position
    Or(if mode == all):
    a tuple contains all data followed by this order:
      train, val, test, count_train, count_val, count_test

    data dictionary: 
      key: brand
      value: list of directory path
    
    count dictionary:
      key: brand
      value: the number of data in the group
    """
    self.mode = mode.lower()
    self.train_p = train_path
    self.test_p = test_path
    self.val_p = val_path
    self.car_brand_list = os.listdir(self.train_p)

  def load(self):
    # Load the data into tuple of dictionaries
    data = defaultdict(list)

    if self.mode == 'train':
      for i, brand in enumerate(self.car_brand_list):
        brand_level_path1 = os.path.join(self.train_p, brand)
        if os.path.isdir(brand_level_path1):
          img_list1 = os.listdir(brand_level_path1)
          for image in img_list1:
            img_path1 = os.path.join(brand_level_path1, image)
            data[brand].append(img_path1)
      count_by_brand = {k: len(data[k]) for k in list(data.keys())}
      self.count_by_brand = count_by_brand

    elif self.mode == 'test':
      for i, brand in enumerate(self.car_brand_list):
        brand_level_path1 = os.path.join(self.test_p, brand)
        if os.path.isdir(brand_level_path1):
          img_list1 = os.listdir(brand_level_path1)
          for image in img_list1:
            img_path1 = os.path.join(brand_level_path1, image)
            data[brand].append(img_path1)
      count_by_brand = {k: len(data[k]) for k in list(data.keys())}
      self.count_by_brand = count_by_brand


    elif self.mode == 'val':
      for i, brand in enumerate(self.car_brand_list):
        brand_level_path1 = os.path.join(self.val_p, brand)
        if os.path.isdir(brand_level_path1):
          img_list1 = os.listdir(brand_level_path1)
          for image in img_list1:
            img_path1 = os.path.join(brand_level_path1, image)
            data[brand].append(img_path1)
      count_by_brand = {k: len(data[k]) for k in list(data.keys())}
      self.count_by_brand = count_by_brand

    elif self.mode == 'all':
      train = defaultdict(list)
      val = defaultdict(list)
      test = defaultdict(list)
      
      for i, brand in enumerate(self.car_brand_list):
        brand_level_path1 = os.path.join(self.train_p, brand)
        img_list1 = os.listdir(brand_level_path1)
        for image in img_list1:
          img_path1 = os.path.join(brand_level_path1, image)
          train[brand].append(img_path1)
      count_by_brand_train = {k: len(train[k]) for k in list(train.keys())}

      for i, brand in enumerate(self.car_brand_list):
        brand_level_path2 = os.path.join(self.val_p, brand)
        img_list2 = os.listdir(brand_level_path2)
        for image in img_list2:
          img_path2 = os.path.join(brand_level_path2, image)
          val[brand].append(img_path2)
      count_by_brand_val = {k: len(val[k]) for k in list(val.keys())}

      for i, brand in enumerate(self.car_brand_list):
        brand_level_path3 = os.path.join(self.test_p, brand)
        img_list3 = os.listdir(brand_level_path3)
        for image in img_list3:
          img_path3 = os.path.join(brand_level_path3, image)
          test[brand].append(img_path3)
      count_by_brand_test = {k: len(test[k]) for k in list(test.keys())}
      self.train = train
      self.val = val
      self.test = test

      self.count_by_brand_train = count_by_brand_train
      self.count_by_brand_test = count_by_brand_test
      self.count_by_brand_val = count_by_brand_val
      return train, val, test, count_by_brand_train, count_by_brand_val, count_by_brand_test

    else:
      print("Invalid Mode! ")
      return
    self.data = data
    return data, count_by_brand
    
  def length(self):
    # Show the data information including the number of data and the number of classes in each dataset
    if self.mode == 'all':
      print('train data: \n Number of label: {}  Number of data: {}\n'.format(len(self.train), sum(list(self.count_by_brand_train.values()))))
      print('val data: \n Number of label: {}  Number of data: {}\n'.format(len(self.val), sum(list(self.count_by_brand_val.values()))))
      print('test data: \n Number of label: {}  Number of data: {}\n'.format(len(self.test), sum(list(self.count_by_brand_test.values()))))
      print("Total Number of data: {}\n".format(sum(list(self.count_by_brand_train.values())) + sum(list(self.count_by_brand_test.values())) + sum(list(self.count_by_brand_val.values()))))
    elif self.mode in {'test', 'val', 'train'}:
      print('{} data: \n Number of label: {}  Number of data: {}\n'.format(self.mode, len(self.data), sum(list(self.count_by_brand.values()))))


def load_to_tensor(data: list, resize_shape: int, batch_size:int):
  """
  Data: a list containing train data, validation data, test data. The order should be followed.
  Resize_shape: the image size you want to modify to
  Batch size: the number images in a batch

  Return a tuple containing train_dataloader, validation_dataloader, test_dataloader and a dictionary with the index and brand information
  """
  train, val, test = data[0], data[1], data[2]
  train_data = []
  train_labels = []
  label_dict = dict()
  c = 0
  for k, v in train.items():
    if k.lower() not in label_dict:
      label_dict[k.lower()] = c
      c += 1
      for i_p in v:
        img = Image.open(i_p)
        img = T.Resize((resize_shape, resize_shape))(img)
        img = T.ToTensor()(img)
        train_data.append(img)
        train_labels.append(label_dict[k.lower()])

  val_data = []
  val_labels = []
  for k, v in val.items():
    for i_p in v:
      img = Image.open(i_p)
      img = T.Resize((resize_shape, resize_shape))(img)
      img = T.ToTensor()(img)
      val_data.append(img)
      val_labels.append(label_dict[k.lower()])

  test_data = []
  test_labels = []
  for k, v in test.items():
    for i_p in v:
        img = Image.open(i_p)
        img = T.Resize((resize_shape, resize_shape))(img)
        img = T.ToTensor()(img)
        test_data.append(img)
        test_labels.append(label_dict[k.lower()])

  train_tensor = torch.stack(train_data).reshape(-1, 3, resize_shape, resize_shape)
  val_tensor = torch.stack(val_data).reshape(-1, 3, resize_shape, resize_shape)
  test_tensor = torch.stack(test_data).reshape(-1, 3, resize_shape, resize_shape)

  train_label_tensor = torch.Tensor(train_labels)
  val_label_tensor = torch.Tensor(val_labels)
  test_label_tensor = torch.Tensor(test_labels)

  train_dataset = TensorDataset(train_tensor, train_label_tensor)
  val_dataset = TensorDataset(val_tensor, val_label_tensor)
  test_dataset = TensorDataset(test_tensor, test_label_tensor)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader, label_dict


def plot_data_distribution(save_p: os.path, count_info: dict, mode:str):
  """
  Plot the data distribution for all classes
  count_info: a dictionary containing the brand and count information(which could be generated using load)
  mode:
  train/val/test: plot data distribution for the specified data group
  """
  fig, ax = plt.subplots(figsize =(25, 25))

  # Horizontal Bar Plot
  ax.barh(list(count_info.keys()), list(count_info.values()))

  # Plot by the order as the one printed above
  ax.invert_yaxis()

  # Add annotations
  for i in ax.patches:
      plt.text(i.get_width()+0.2, i.get_y()+0.5,
              str(round((i.get_width()), 2)),
              fontsize = 10, fontweight ='bold',
              color ='grey')
  ax.set_title("Car {} data distribution by brand".format(mode))
  ax.set_xlabel("Count")
  ax.set_ylabel("Brand Name")
  plt.savefig(save_p)

def plot_fig(save_path: os.path, train_loss: list, val_loss: list, accu = False):
  """
  save_path: where the figure to be saved
  train_loss: a list containing the train losses
  val_loss: a list containing the validation losses
  accu: whether to plot the accuracy plot or loss plot
  """
  plt.figure(figsize=(10,8))
  if accu:
    # Plot accuracy plot
    plt.plot(train_loss, label= 'Train_Accuracy')
    plt.plot(val_loss, label= 'Val_Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.join(save_path, "figures/train_val_accu.png"))
  
  else:  
    # Plot loss
    plt.plot(train_loss, label= 'Train_loss')
    plt.plot(val_loss, label= 'Val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.join(save_path, "figures/train_val_loss.png"))

  plt.show()

