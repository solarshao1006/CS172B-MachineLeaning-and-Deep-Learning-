from helpers import load_data, plot_data_distribution
from data_augmentation import data_augment
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


img_p = '/Users/apple/Desktop/study/Coding/CS172B/Project/Splitted_data_set_original/Splited_dataset/train/Honda_civic_2015/17872.jpg'
img = Image.open(img_p)
blurr = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
color = T.ColorJitter(brightness=.5, hue=.3)(img)
sharpness_adjuster = F.adjust_sharpness(sharpness_factor=5)(img)
rotater = T.RandomRotation(degrees=(-30,30))(img)
