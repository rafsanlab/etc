"""
Created on Fri 28 Apr 2023
Galway, Ireland.

@author: Rafsanjani @rafsanlab

"""

import torch.nn as nn

class ConvAutoencoder(nn.Module):
  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    
    # Encoder
    self.enc1 = nn.Conv2d(467, 256, 3, stride=2, padding=1)
    self.bnm1 = nn.BatchNorm2d(256)
    self.enc2 = nn.Conv2d(256, 128, 3, stride=2, padding=1)
    self.bnm2 = nn.BatchNorm2d(128)
    self.enc3 = nn.Conv2d(128, 64, 3, stride=2, padding=1)
    self.bnm3 = nn.BatchNorm2d(64)

    # Decoder
    self.dec1 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1)
    self.bnm4 = nn.BatchNorm2d(128)
    self.dec2 = nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1, output_padding=1)
    self.bnm5 = nn.BatchNorm2d(256)
    self.dec3 = nn.ConvTranspose2d(256, 3, 3, stride=2, padding=1, output_padding=1)
    
    self.act_fn = nn.ReLU()
    self.act_fn_final = nn.Sigmoid()
      
  def forward(self, x):
    # Encoder
    x = self.act_fn(self.bnm1(self.enc1(x)))
    x = self.act_fn(self.bnm2(self.enc2(x)))
    x = self.act_fn(self.bnm3(self.enc3(x)))

    # Decoder
    x = self.act_fn(self.bnm4(self.dec1(x)))
    x = self.act_fn(self.bnm5(self.dec2(x)))
    x = self.act_fn_final(self.dec3(x))
    return x*255

import matplotlib.ticker as tck
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, criterion, epochs,
                filenamedir='img.svg', dpi=100):
  """
  Plot train and test losses.
  """
  fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
  ax.plot(range(1, epochs+1), train_losses, linewidth=2, label='Train')
  ax.plot(range(1, epochs+1), test_losses, linewidth=2, label='Test', color='red')
  ax.title.set_text('Losses')
  ax.set_xlabel('Epochs')
  ax.set_ylabel(str(criterion))
  ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
  ax.minorticks_on()
  ax.grid()
  ax.legend()
  plt.autoscale(enable=True, axis='x')
  plt.savefig(filenamedir)
  plt.show()

import numpy as np

## helper functions

def model_output(data_loader, model):
  # get one batch of test images
  dataiter = iter(data_loader)
  refs, imgs = next(dataiter)

  # get sample outputs
  model = model.to('cpu')
  model.eval()
  outs = model(imgs)
  outs = outs.detach()
  refs = refs.detach()
  return refs, outs

def imshowT(img): plt.imshow(np.transpose(img, (1, 2, 0)))

def check_array(arr):
  print('Shape: {}\t | Min values: {:.6f} | Max values: {:.6f}'
        .format(arr.shape, np.min(arr), np.max(arr))
        )