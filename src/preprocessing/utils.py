import numpy as np
from PIL import Image
import random

def resize_image(img, size=(512, 512)):
    return img.resize(size, Image.BILINEAR)

def resize_mask(mask, size=(512, 512)):
    return mask.resize(size, Image.NEAREST)

def random_rotation(img, mask):
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    
    img = img.rotate(angle)
    mask = mask.rotate(angle)
    
    return img, mask