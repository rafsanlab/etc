"""
Created on Fri 27 Jan 2023
Dublin, Ireland.

@author: Rafsanjani @rafsanlab

"""
import numpy as np
import os
import shutil
import pathlib
import re
import urllib.request
import zipfile


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


def get_fname(filepath, pattern:str, n=3, separator='_'):
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
    filepath = pathlib.Path(filepath)
    filename = filepath.parts[-1] 
    filename_parts = re.split(pattern, str(filename))
    modified_filename = ''
    for i in range(n):
        modified_filename += filename_parts[i]
        if i != n-1:
            """ this condition check last filename_parts so that
                no separator at the modified_filename end """
            modified_filename += separator
    # modified_filename = f"{filename_parts[0]}_{filename_parts[1]}_{filename_parts[2]}"
    
    return modified_filename
  

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


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


def create_dir(path:str, verbose=True):
    """
    Function to create directory if not exist.

    Args:
        path(str): path directory.
        verbose(bool): output condition status.
    
    """
    
    path = pathlib.Path(path)
    if os.path.exists(path) == False:
        os.makedirs(path)
        if verbose == True: print('Path created: \t', path)
    elif os.path.exists(path) == True:
        if verbose == True: print('Path already exist.')


def create_project_dir(project_dir='', sub_dirs=[], verbose=True, return_dict=False):
    """
    A function to create a main dir, then sub dirs inside.
    Optional to return dict to the paths usinf sub_dirs as keys.

        """
    # Create main directory
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir, project_dir)

    # Create subdirectories
    if len(sub_dirs) > 0:
        dirs = {sub_dir: os.path.join(project_dir, sub_dir) for sub_dir in sub_dirs}

    # Create directories
    dirs['project_dir'] = project_dir
    for dir_k, dir_v in dirs.items():

        if os.path.exists(dir_v) == False:
            os.makedirs(dir_v)
            if verbose == True: print('Path created: \t', dir_v)
        elif os.path.exists(dir_v) == True:
            if verbose == True: print('Path exist: \t', dir_v)

    # Return dict
    if return_dict == True:
        return dirs
    

def copycut_contents(sourcedir, targetdir, verbose=True):
    """
    A function to move contents of source folder to target folder.

    """
    
    if not os.path.exists(sourcedir):
        print(f"sourcedir not exist : '{sourcedir}")
        return
    
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
        print(f"Created directory : '{targetdir}")
    
    base_dir = os.path.basename(sourcedir)

    items = os.listdir(sourcedir)

    for item in items:
        
        source_path = os.path.join(sourcedir, item)
        target_path = os.path.join(targetdir, base_dir, item)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.move(source_path, target_path)
        if verbose:
            print(f"Moved '{item}' ->>> '{targetdir}'")


def move_folders_contents(source_directory, target_directory, verbose=True):
    """
    function to move contents of source folder to target folder
    
    Example:
        >>> dir1/content...n
        >>> dir2/content...n

    """
    print('This function will be remove, please use copycut_contents()')
    # Check if both source and target directories exist
    if not os.path.exists(source_directory):
        
        print(f"source_directory not exist : '{source_directory}")
        return
    
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Create directory : '{target_directory}")

    # Get the list of items (folders and files) in the source directory
    items = os.listdir(source_directory)

    for item in items:
        
        # Construct the full path of the item
        source_path = os.path.join(source_directory, item)
        target_path = os.path.join(target_directory, item)
        shutil.move(source_path, target_path)
        if verbose:
            print(f"Moved '{item}' to '{target_directory}'")

        # # Check if the item is a directory
        # if os.path.isdir(source_path):
        #     # Construct the destination path in the target directory
        #     target_path = os.path.join(target_directory, item)

        #     # Move the entire directory to the target directory
        #     shutil.move(source_path, target_path)
        #     if verbose:
        #         print(f"Moved '{item}' to '{target_directory}'")


def zip_folder(source_folder, output_filename):
    """
    Zip target folder and its contents.

    # Example:
        >>> source_folder = '/content/patches64/train'
        >>> output_filename = '/content/patches64/train' # '/train' here will be 'train.zip'
        >>> zip_folder(source_folder, output_filename)  
    """
    shutil.make_archive(output_filename, 'zip', source_folder)


def unzip_folder(zip_filename, extract_folder):
    """
    Unzip target folder to target output folder.

    # Example:
        >>> zip_filename = '/content/metadata.zip'
        >>> extract_folder = '/content/metadata'
        >>> unzip_folder(zip_filename, extract_folder)
    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


# ------------------------------------------------------------------------
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
    filepath = pathlib.Path(filepath)
    filename = filepath.parts[-1] 
    filename_parts = re.split(pattern, str(filename))
    modified_filename = ''
    for i in range(n):
        modified_filename += filename_parts[i]
        if i != n-1:
            """ this condition check last filename_parts so that
                no separator at the modified_filename end """
            modified_filename += separator
    # modified_filename = f"{filename_parts[0]}_{filename_parts[1]}_{filename_parts[2]}"
    
    return modified_filename

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
