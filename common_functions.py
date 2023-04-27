"""
Created on Fri 27 Jan 2023
Dublin, Ireland.

@author: Rafsanjani @rafsanlab

"""

from pathlib import Path
import numpy as np
import os
import pathlib
import re

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
#   for i in data_path: print(i); break
    print(f'Total data paths: {len(data_path)}')
  else:
    pass
  
  return data_path

def createdir(pathx:str, verbose=True):
  """
  Function to create directory.

  Return:
  ------
  pathx = the PosixPath of the created directory

  """
  pathx = Path(pathx)
  if os.path.exists(pathx) == False:
    os.makedirs(pathx)
    if verbose == True: 
      print('Path created.')
  elif os.path.exists(pathx) == True:
    if verbose == True:
      print('Path already exist.')
#   return pathx

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

def getFilename(filepath:pathlib.PosixPath, split:bool=True, split_items:str='[ ]'):
  """
  Return the filename at the end of path, with optional splitting.
  
  """
  filename = filepath.parts[-1]
  if split == True:
    splitted = re.split(split_items, filename)
    return splitted
  elif split == False:
    return filename

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
