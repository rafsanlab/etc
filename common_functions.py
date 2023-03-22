"""
Created on Fri 27 Jan 2023
Dublin, Ireland.

@author: Rafsanjani @rafsanlab

"""

from pathlib import Path
import numpy as np

def getdata(PATH:str, FILE_FORMAT:str, VERBOSE:bool):
  '''
  Get data from path with spesific filetypes.

  Parameters
  ----------
  PATH : 'c/path/to/folder'
  FILE_FORMAT : '.jpg' or '.txt'
  VERBOSE : True or False
  
  Returns
  -------
  paths : an array containing list of directories to data

  '''
  # parameters
  PATH = Path(PATH)
  FILE_FORMAT = FILE_FORMAT
  VERBOSE = VERBOSE

  # execution
  lookfor = '*' + FILE_FORMAT
  data_path = list(PATH.glob(lookfor))
  data_path = sorted(data_path)
  if VERBOSE==True:
    for i in data_path: print(i); break
    print(f'Total data paths: {len(data_path)}')
  else:
    pass
  
  return data_path


def imgInfo(img):
  '''
  Function to print image (np.array) details
  like type, shape, min and max values.

  Parameters
  ----------
  img : np array

  Returns
  -------
  print of np array details
  
  '''
  a = img.dtype
  b = img.shape
  c = np.min(img)
  d = np.max(img)
  print(f'dtype:{a} | shape:{b} | min:{c} | max:{d}')

def incontrast(img, lv, up, x=1):
  """
  Increate contrast by clipping lower
  and upper value of an image.

  Parameters
  ----------
  img : np.array of an image
  lv  : int of lower value to clip
  up  : int of upper value to clip
  x   : int of multiplication i.e: 255 for RGB

  Return
  ------
  img : np.array of an image

  """
  lv, up, x = lv, up, x
  minval = np.percentile(img, lv)
  maxval = np.percentile(img, up)
  img = np.clip(img, minval, maxval)
  img = ((img-minval)/(maxval-minval)) * x
  return img