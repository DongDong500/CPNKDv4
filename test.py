import os
import torch

gpus = "2,7"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

if __name__ == "__main__":
    
    
    """devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (devices, str(gpus)))

    torch.rand((10,10)).to(devices)
    torch.rand((10,10)).to(devices)
    print(os.path.abspath(__file__))
    print(os.path.dirname( os.path.abspath(__file__) ))

    for i in range(10):
        torch.rand((10,10)).to(devices)"""

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    _dir = '/data1/sdi/datasets/CPN_all'

    images = [os.path.join(_dir, 'Images', x) for x in os.listdir(os.path.join(_dir, 'Images'))]
    masks = [os.path.join(_dir, 'Masks', x) for x in os.listdir(os.path.join(_dir, 'Masks'))]

    print(f'len: {len(images)}')

    X, Y = [], []

    for ma in masks:
        m = Image.open(ma).convert('L')
        print(type(m))


