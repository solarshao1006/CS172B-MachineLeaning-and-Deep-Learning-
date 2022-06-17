
import torchvision.transforms as T
from PIL import Image

"""
This .py file contains related functions used for data augmentation
"""

def blur(times: int, image):
    """
    Blur the given image(PIL format) by the number of times specify by 'times' parameter
    Return the new image in PIL image format and the suffix as list
    """
    blurr = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    img_p =  [blurr(image) for _ in range(times)]
    name_p = ['blur-0{}.jpg'.format(i) for i in range(1,times+1)]
    return img_p, name_p

def rotation(times: int, image):
    """
    Rotate the given image(PIL format) by the number of times specify by 'times' parameter
    Return the new image in PIL image format and the suffix as list
    """
    rotater = T.RandomRotation(degrees=(-30,30))
    img_p = [rotater(image) for _ in range(times)]
    name_p = ['rotate-0{}.jpg'.format(i) for i in range(1,times+1)]
    return img_p, name_p

def color_jitter(times: int, image):
    """
    Modify the color the given image(PIL format) by the number of times specify by 'times' parameter
    Return the new image in PIL image format and the suffix as list
    """
    jitter = T.ColorJitter(brightness=.5, hue=.3)
    img_p = [jitter(image) for _ in range(times)]
    name_p = ['color_jitter-0{}.jpg'.format(i) for i in range(1,times+1)]
    return img_p, name_p

def sharpen(times: int, image):
    """
    Sharpenn the given image(PIL format) by the number of times specify by 'times' parameter
    Return the new image in PIL image format and the suffix as list
    """
    sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=5)
    img_p = [sharpness_adjuster(image) for _ in range(times)]
    name_p = ['sharpen-0{}.jpg'.format(i) for i in range(1,times+1)]
    return img_p, name_p


def data_augment(times, path_lst, selected_number):
    """
    Times: the number of augment performed
        20 times: 4 blur, 4 rotation, 8 color jitter, 4 sharpen
        10 times: 3 blur, 2 rotation, 3 color jitters, 2 sharpen
        8 times: 2 blur, 2 rotation, 2 color jitters, 2 sharpen
        6 times: 3 blur, 2 rotation, 1 color jitters
        5 times: 2 blur, 2 rotation, 1 color jitters
        4 times: 2 blur, 2 color jitters
        3 times: 1 blur, 2 color jitters
        2 times: 1 blur, 1 color jitters
    
    path_lst:
    the image directory

    selected number: 
    the number of images to be selected and perform augmentation
    """
    path_set = set(path_lst)
    if times == 20:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 4 times
            m_img +=  blur(4, original_img)[0]
            new_path += blur(4, original_img)[1]

            # Rotation 4 times
            m_img += rotation(4, original_img)[0]
            new_path += rotation(4, original_img)[1]

            # Color jitter 8 times
            m_img += color_jitter(8, original_img)[0]
            new_path += color_jitter(8, original_img)[1]

            # Sharpen 4 times
            m_img += sharpen(4, original_img)[0]
            new_path += sharpen(4, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)
    elif times == 10:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 3 times
            m_img +=  blur(3, original_img)[0]
            new_path += blur(3, original_img)[1]

            # Rotation 2 times
            m_img += rotation(2, original_img)[0]
            new_path += rotation(2, original_img)[1]

            # Color jitter 3 times
            m_img += color_jitter(3, original_img)[0]
            new_path += color_jitter(3, original_img)[1]

            # Sharpen 2 times
            m_img += sharpen(2, original_img)[0]
            new_path += sharpen(2, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)
    elif times == 8:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 2 times
            m_img +=  blur(2, original_img)[0]
            new_path += blur(2, original_img)[1]

            # Rotation 2 times
            m_img += rotation(2, original_img)[0]
            new_path += rotation(2, original_img)[1]
            
            # Color jitter 2 times
            m_img += color_jitter(2, original_img)[0]
            new_path += color_jitter(2, original_img)[1]

            # Sharpen 2 times
            m_img += sharpen(2, original_img)[0]
            new_path += sharpen(2, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)
    elif times == 6:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 3 times
            m_img +=  blur(3, original_img)[0]
            new_path += blur(3, original_img)[1]

            # Rotation 2 times
            m_img += rotation(2, original_img)[0]
            new_path += rotation(2, original_img)[1]

            # Color jitter 1 times
            m_img += color_jitter(1, original_img)[0]
            new_path += color_jitter(1, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)
    elif times == 5:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 2 times
            m_img +=  blur(2, original_img)[0]
            new_path += blur(2, original_img)[1]

            # Rotation 2 times
            m_img += rotation(2, original_img)[0]
            new_path += rotation(2, original_img)[1]

            # Color jitter 1 times
            m_img += color_jitter(1, original_img)[0]
            new_path += color_jitter(1, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)

    elif times == 4:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 2 times
            m_img +=  blur(2, original_img)[0]
            new_path += blur(2, original_img)[1]

            # Color jitter 2 times
            m_img += color_jitter(2, original_img)[0]
            new_path += color_jitter(2, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)

    elif times == 3:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 1 times
            m_img +=  blur(1, original_img)[0]
            new_path += blur(1, original_img)[1]
            
            # Color jitter 2 times
            m_img += color_jitter(2, original_img)[0]
            new_path += color_jitter(2, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)

    elif times == 2:
        for _ in range(selected_number):
            original_img_path = path_set.pop()
            original_img = Image.open(original_img_path)
            m_img = []
            new_path = []

            # Blur 1 times
            m_img +=  blur(1, original_img)[0]
            new_path += blur(1, original_img)[1]
            
            # Color jitter 1 times
            m_img += color_jitter(1, original_img)[0]
            new_path += color_jitter(1, original_img)[1]

            for img, mpath in zip(m_img, new_path):
                img.save(original_img_path[:-4] + mpath)
if __name__ == "__main__":
    from helpers import load_data
    import random
    import os
    loader = load_data('ALL')
    # Load data
    train, val, test, count_by_brand_train, count_by_brand_val, count_by_brand_test = loader.load()
    
    # Sample data augmentation code:
    # Augment groups less than 80  by applying three data augmentation using 2 images
    for k, v in count_by_brand_test.items():
        if v < 80:
            data_augment(3, test[k], 2)

    # Remove exceeded data
    for k,v in count_by_brand_test.items():
        if v > 80:
            res = v - 80
            for _ in range(res):
                c = random.choice(range(len(test[k])))
                os.remove(test[k][c])
                loader = load_data('test')
                test, count_by_brand_test = loader.load()
        
