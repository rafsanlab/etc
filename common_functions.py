"""
Created on Fri 27 Jan 2023
Dublin, Ireland.

@author: Rafsanjani @rafsanlab

"""

import numpy as np
import os
import pathlib
import re
import urllib.request


def get_data(path:str, file_type:str, verbose:bool):
  """
  Get files from path with sepsific file types.

  Args:
        path(str): string path containing the files
        file_type(str): file type such as '.png'
        verbose(bool): condition of output summary
  Return:
        paths(list): list of paths
  """

  path = pathlib.Path(path)
  lookfor = '*' + file_type
  paths = list(path.glob(lookfor))
  paths = sorted(paths)
  if verbose == True:
    print(f'Total paths: {len(paths)}')
  else:
    pass
    
  return paths


def get_filename(filepath, pattern:str, n=3, separator='_'):
    """
    Modify the filename from a_b_c_n.d to a_b_c (if pattern='[._]' and n=3).
    
    Args:
        filepath (str): File path.
        pattern (str): re.split argument i.e: '[._]'.
        n (int): First number of names to be keep.
        separator (str): Name separator for final output.
    
    Return:
        str: Modified filename.
    """

    filename = filepath.parts[-1] 
    filename_parts = re.split(pattern, filename)
    modified_filename = ''
    for i in range(n):
        modified_filename += filename_parts[i]
        if i != n:
            """ this condition check last filename_parts so that
                no separator at the modified_filename end """
            modified_filename += separator
    # modified_filename = f"{filename_parts[0]}_{filename_parts[1]}_{filename_parts[2]}"
    
    return modified_filename


def get_fonts_in_Colab():
    """ Allow user to install fonts from the URL into Colab. This give option to use custom font
        especially in Matplotlib. Fonts will be rename to fit Colab standard. """
  
    font_dir = '/usr/share/fonts/truetype/san-serif'
    os.makedirs(font_dir, exist_ok=True)
    fonts = [
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIAL.TTF', 'Arial-Regular.ttf'),
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIALBD.TTF', 'Arial-Bold.ttf'),
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIALI.TTF', 'Arial-Italic.ttf')
    ]
    for url, filename in fonts:
        font_path = os.path.join(font_dir, filename)
        urllib.request.urlretrieve(url, font_path)











# old version of codes :

from pathlib import Path

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

import os
import urllib.request

def getFonts_inColab():
    """
    use to get fonts based on URLs, renamed and put into a font family folder in colab
    
    """
    font_dir = '/usr/share/fonts/truetype/san-serif'
    os.makedirs(font_dir, exist_ok=True)

    fonts = [
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIAL.TTF', 'Arial-Regular.ttf'),
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIALBD.TTF', 'Arial-Bold.ttf'),
        ('https://github.com/rafsanlab/etc/raw/main/Fonts/Arial/ARIALI.TTF', 'Arial-Italic.ttf')
    ]

    for url, filename in fonts:
        font_path = os.path.join(font_dir, filename)
        urllib.request.urlretrieve(url, font_path)
      

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
