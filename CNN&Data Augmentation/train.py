
import os
import torch
from torch.autograd import Variable
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

def train(train_process_p: os.path, validation_process_p: os.path, device, model, optimizer, criterion, num_epoch: int, train_loader:DataLoader, val_loader: DataLoader, batch_size:int):
    """
    Params:
    train_process_p: a directory where final training process txt file including epoch, loss and accuracy will be saved
    val_process_p: a directory where final validation process txt file including epoch, loss and accuracy will be saved

    device: sepcify the device you want you code to be run on(GPU or CPU)

    model: the model you are using
    optimizer: the optimizer for training process
    criterion: the loss function for training process
    num_epoch: the number of epoch

    train_loader: a dataloader contianing the train data
    val_loader: a dataloader contianing the validation data

    batch_size: your data batch_size
    """
    
    total_train_loss = []
    total_val_loss = []
    patience = 4
    triggertimes = 0
    last_loss =100
    early_stop = False
    train_acc = []
    val_acc = []

    for epochs in range(num_epoch):
      cnn.train()
      train_loss = []
      total_train_label = []
      total_train_predictions = []
      
      print("Epoch: {}\tStart Training".format(epochs+1))
      for idx, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        img = Variable(img).to(device)
        label = Variable(label).type(torch.LongTensor).to(device)
        scores = cnn(img)
        
        loss = criterion(scores, label)
        train_loss.append(loss.item() * len(label.cpu()))

        loss.backward()       
        optimizer.step()

        _, predictions = scores.max(dim=1)
        total_train_predictions += predictions.cpu()
        total_train_label += label.cpu()

        print('Iter: [{}/{}]\t Epoch: [{}/{}]\t Loss: {:.6f}\t Acc: {:.6f}'.format(idx+1, len(train_loader), epochs+1, num_epoch,
                                                                        loss.item(),
                                                                        metrics.accuracy_score(label.cpu(), predictions.cpu()))) 
      train_loss_per_epoch = sum(train_loss) / (len(train_loader) * batch_size)
      total_train_loss.append(train_loss_per_epoch)

      train_acc_per_epoch =  metrics.accuracy_score(total_train_label, total_train_predictions)
      train_acc.append(train_acc_per_epoch)

      with open(train_record_p, 'a') as f:
        f.write('Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}\n'.format(epochs+1, train_loss_per_epoch, train_acc_per_epoch))
      print('Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epochs+1, train_loss_per_epoch, train_acc_per_epoch))
      

      cnn.eval()
      val_loss = []
      total_val_label = []
      total_val_predictions = []
      loss_total = 0

      print("Epoch: {}\tStart Validation".format(epochs+1))
      with torch.no_grad():
        for idx, (img, label) in enumerate(val_loader):
          img = Variable(img).to(device)
          label = Variable(label).type(torch.LongTensor).to(device)
          scores = cnn(img)
          loss = criterion(scores, label)
          val_loss.append(loss.item() * len(label.cpu()))
          loss_total += loss.item()

          _, predictions = scores.max(dim=1)
          total_val_predictions += predictions.cpu()
          total_val_label += label.cpu()
          
          print('Iter: [{}/{}]\t Epoch: [{}/{}]\t Loss: {:.6f}\t Acc: {:.6f}'.format(idx+1, len(val_loader), epochs+1, num_epoch,
                                                                        loss.item(),
                                                                        metrics.accuracy_score(label.cpu(), predictions.cpu()))) 
      val_loss_per_epoch = sum(val_loss) / (len(val_loader) * batch_size)
      total_val_loss.append(val_loss_per_epoch)
      val_acc_per_epoch = metrics.accuracy_score(total_val_label, total_val_predictions)
      val_acc.append(val_acc_per_epoch)

      with open(val_record_p, 'a') as f:
        f.write('Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}\n'.format(epochs+1, val_loss_per_epoch, val_acc_per_epoch))
      print('Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epochs+1, val_loss_per_epoch, val_acc_per_epoch))
      curr_loss = loss_total / len(val_loader)
      if curr_loss > last_loss:
        triggertimes += 1
        print('Trigger Times: ', triggertimes)
        if triggertimes >= patience:
          print("Early stopping! \nEpoch: {}\tTraining Loss: {}\tTraing Accuracy: {}\nValidation Loss: {}\tValidation Accuracy: {}".format(epochs+1, total_train_loss, train_acc_per_epoch, total_val_loss, val_acc_per_epoch))
          early_stop = True
      else:
        triggertimes = 0
      last_loss = curr_loss
      if early_stop:
        break

    return total_train_loss, total_val_loss, train_acc, val_acc
