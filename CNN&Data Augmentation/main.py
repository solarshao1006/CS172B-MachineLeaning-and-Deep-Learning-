from pyexpat import model
import helpers
from train import train
from CNN import CNN
import torch.nn as nn
import torch.optim as optim
import torch
import sklearn.metrics as metrics
from test import test

loader = helpers.load_data('ALL', '/Users/apple/Desktop/study/Coding/CS172B/Project/Splitted_data_set_original/Splited_dataset/test')
train_, val_, test_, count_by_brand_train, count_by_brand_val, count_by_brand_test = loader.load()
helpers.plot_data_distribution('/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/test_distribution', count_by_brand_train, 'Test')
helpers.plot_data_distribution('/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/train_distribution', count_by_brand_test, 'Train')
helpers.plot_data_distribution('/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/val_distribution', count_by_brand_val, 'Validation')


device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epoch = 200
batch_size = 50
learning_rate = 0.005

# Load data
train_loader, val_loader, test_loader, car_brand_list = helpers.load_to_tensor([train_, val_, test_], 128, batch_size)
num_classes = len(car_brand_list)

cnn = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

train_record_p = '/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/train_result.txt'
val_record_p = '/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/val_result.txt'
loss_plt_p = '/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/loss_plt.png'
acc_plt_p = '/Users/apple/Desktop/study/Coding/CS172B/Project/Final Result/acc_plt.png'

total_train_loss, total_val_loss, train_acc, val_acc = train(train_record_p, val_record_p, device, cnn, optimizer, criterion, num_epoch, train_loader, val_loader, batch_size)
helpers.plot_fig(loss_plt_p, total_train_loss, total_val_loss)
helpers.plot_fig(train_acc, val_acc, accu=True)

test(device, test_loader, model)
