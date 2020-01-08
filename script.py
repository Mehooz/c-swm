import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)
if __name__=='__main__':
    img = Image.open('mask_1.jpg')
    img =img.convert('1')
    img.save('mask_one_hot.jpg')
