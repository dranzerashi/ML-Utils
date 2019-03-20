import numpy as np
import cv2

class Gram:
    def __init__(self, img_path, label):
        self.img_path = img_path
        self.label = label
        self.width = None
        self.height = None
    def load_img(self):
        img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.width = img.shape[1]
        self.height = img.shape[0]
        return img

class NGram:
    def __init__(self):
        self.grams = []
        self.img = None
        self.label = ""

    def add_gram(self, gram):
        self.grams.append(gram)
        self.generate_n_gram()
        
    def generate_n_gram(self):
        imgs = [i.load_img() for i in self.grams]
        labels = [i.label for i in self.grams]
        self.img = np.concatenate(imgs, axis=1)
        self.label = ''.join(labels)
    
    def bbox(self, index):
        start_x = 0
        start_y = 0
        for i in range(index):
            start_x += self.grams[i].width
        end_x = start_x + self.grams[index].width
        end_y = self.grams[index].height
        return ((start_x, start_y), (end_x, end_y))