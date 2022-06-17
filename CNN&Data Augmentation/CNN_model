# CNN Model
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, num_classes=...):
    super().__init__()
  
    self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    
    self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

    self.layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding =1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    
    self.layer4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

    self.layer5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(512),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(512*8*8, 128)
    self.fc2 = nn.Linear(128, num_classes)

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.flatten(x)
    x = self.relu(x)

    x = self.fc1(x)
    x = self.relu(x)

    x = self.fc2(x)
    x = self.softmax(x)
    return x
