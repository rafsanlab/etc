"""
Created on Mon 13 Feb 2023
Dublin, Ireland.

@author: Rafsanjani @rafsanlab

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

def thres_otsu(img, blur_kernel=(3,3), tval=0, maxval=255):
  '''
  Applies otsu thresholding
  
  Parameters
  ----------
  img         : array of image
  blur_kernel : gaussian blur kernel
  tval        : thresholding value
  maxval      : thresholded value

  Returns
  -------
  thres       : img array

  '''
  img, blur_kernel, tval, maxval = img, blur_kernel, tval, maxval

  # checking img input
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  elif len(img.shape) == 2:
    img = img
  else:
    print('Error: Image input invalid.')
    return None
  
  # img > blur > Otsu
  img = cv2.GaussianBlur(img, blur_kernel, 0)
  thresh = cv2.threshold(img, tval, maxval, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  return thresh


def rm_debris(img, X1=0.01):
  '''
  Remove small particles in an 2D image
  
  Parameters
  ----------
  img     : an array of 2D image
  X1      : multiplier of average area of the image
            (the smaller the X1 value,
            the bigger the particle size to be remove)

  Returns
  -------
  thresh  : thresholded image array
  
  '''

  thresh, X1 = img, X1

  # checking img input
  if len(img.shape) != 2:
    print('Error: Image input invalid.')
    return None

  # determine average area
  average_area = [] 
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w * h
      average_area.append(area)
  average = sum(average_area) / len(average_area)

  # remove 'debris'
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      area = cv2.contourArea(c)
      if area < average * X1:
          cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
  return thresh


def rm_holes(img, X1=0.1, holes_kernel=(5,5), iterations=2):
  '''
  Remove holes from an 2D image array.

  Parameters
  ----------
  img           : an array of 2D image
  X1            : multiplier of average area size
  holes_kernel  : size of holes to be remove
  interations   : number of iterations 

  Returns
  -------
  close         : image array

  '''
  thresh, X1, iterations = img, X1, iterations

  # checking img input
  if len(img.shape) != 2:
    print('Error: Image input invalid.')
    return None

  # determine average area
  average_area = [] 
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w * h
      average_area.append(area)
  average = sum(average_area) / len(average_area)

  # remove 'holes'
  cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      area = cv2.contourArea(c)
      if area < average * X1:
          cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
  
  # Morph close and invert image
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, holes_kernel)
  close = cv2.morphologyEx(
      thresh,cv2.MORPH_CLOSE,
      kernel, iterations=iterations
      )
  
  return close


def getcoor(img):
  """
  Get the top, bottom, left and right coordinates of
  an image (masked 1 and 0 image).
  
  Parameters
  ----------
  img   : np array of an 1D masked image (1 and 0) 

  Return
  -------
  results : a list of the 4 coordinates (top, bot, left, right)

  """
  img = img
  w, h = img.shape[0], img.shape[1]
  coorx, coory = [], []

  # find coor x
  for i in range(w):

    arr = img[i,:]
    minv = np.min(arr)
    maxv = np.max(arr)

    if maxv - minv != 0:
      x = np.argmin(arr)
      x = [x, i]
      coorx.append(x)

  # find coor y
  for i in range(h):

    arr = img[:,i]
    minv = np.min(arr)
    maxv = np.max(arr)

    if maxv - minv != 0:
      x = np.argmin(arr)
      x = [i, x]
      coory.append(x)

  # assemble the coor
  ptv1, ptv2 = coorx[0], coorx[-1]
  pth1, pth2 = coory[0], coory[-1]
  results = [ptv1, ptv2, pth1, pth2]

  return results


def imgs_jaccard(img1, img2, average='weighted'):
  '''
  Find the jaccard score of 2 2D images

  Parameters
  ----------
  img1  : an array of first img
  img2  : an array of seconnd img
  average : paramater of jaccard_score()

  Return
  -----
  scoe  : float of jaccard score

  '''
  img1, img2, average = img1, img2, average
  img1 = img1.astype(int)
  img2 = img2.astype(int)
  img1 = img1.reshape(img1.shape[0] * img1.shape[1])
  img2 = img2.reshape(img2.shape[0] * img2.shape[1])
  score = jaccard_score(img1, img2, average=average)
  return score


def plot1x2(img1, img2):
  '''
  Plot image side by side
  
  Parameters
  ----------
  img1  : an array of first img
  img2  : an array of second img

  Return
  -----
  plt.show()

  '''
  X, Y, dpi, n1 = 1, 2, 90, 0
  _, axs = plt.subplots(X, Y, dpi=dpi)
  axs[0].imshow(img1, cmap='viridis')
  axs[1].imshow(img2, cmap='coolwarm')
  axs[0].title.set_text('')
  axs[1].title.set_text('')
  for i in range(Y): axs[i].title.set_size(n1)
  #for i in range(Y): axs[i].set_axis_off()
  # for ax in axs:
  #     ax.get_xaxis().set_ticks([])
  #     ax.get_yaxis().set_ticks([])
  plt.tight_layout()
  plt.show;


def plot_overlapped(img1, img2):
  '''
  Plot 2 images overlapped
  
  Parameters
  ----------
  img1  : an array of first img
  img2  : an array of second img

  Return
  -----
  plt.show()

  '''
  dpi = 90
  plt.imshow(img1, cmap='viridis', alpha=0.5)
  plt.imshow(img2, cmap='coolwarm', alpha=0.3)
  plt.gca().get_xaxis().set_visible(False)
  plt.gca().get_yaxis().set_visible(False)
  plt.gcf().set_dpi(dpi)
  plt.tight_layout()
  plt.show;

def plot1x3(img1, img2, img3, show=True, save=False, filename='1x3.png',
            pts_src=None, pts_ref=None):
  '''
  Plot 3 subplots where the third subplot
  is an overlapped between between the third image
  and the second image.
  
  Parameters
  ----------
  img1  : an array of first img
  img2  : an array of second img
  img3  : an array of third img to be overlapped with img2
  show  : bool either to plot or not
  save  : bool either to save or not
  filename : str of image name with img format
  pts_src  : [[x1,y1], [x2,y2]] coordinates to mark in img1
  pts_ref  : [[x1,y1], [x2,y2]] coordinates to mark in img2

  Return
  ------
  plt.show()

  '''
  X, Y, dpi, n1 = 1, 3, 100, 0
  #pts_src, pts_ref = pts_src, pts_ref 
  #---
  show, save, filename = show, save, filename
  _, axs = plt.subplots(X, Y, dpi=dpi, figsize=(9,3))
  #---
  axs[0].imshow(img1, cmap='coolwarm')
  axs[1].imshow(img2, cmap='Spectral')
  axs[2].imshow(img3, cmap='bwr', alpha=0.8)
  axs[2].imshow(img2, cmap='Spectral', alpha=0.5)
  
  for pts in pts_src:
    x = int(pts[0])
    y = int(pts[1])
    axs[0].plot(x, y, marker='x', color="white", linewidth=0.5)

  for pts in pts_ref:
    x = int(pts[0])
    y = int(pts[1])
    axs[1].plot(x, y, marker='x', color="white", linewidth=0.5)

  for i in range(Y): axs[i].title.set_size(n1)
  for i in range(Y): axs[i].set_axis_off()
  plt.gcf().set_dpi(dpi)
  plt.tight_layout()
  #---
  if show != True:
    plt.ioff()
  else:
    plt.show;
  #---
  if save == True:
    plt.savefig(filename, transparent=True, dpi=dpi)
    plt.close()
  else:pass

def plotRxC(nrows=1, ncols=2, dpi=120, figsize=(9,3),
            imgs=[], titles=[], title_size=10,
            show=True, save=False, filename='rxc.png'):

  '''
  Plot X number of rows and columns subplot.
  
  Parameters
  ----------
  nrows = number of rows (1)
  ncols = number of columns
  dpi = dpi value of the figure
  figsize = figure size
  imgs  = list of image array to be plotted
  titles  = list of titles in string
  title_size  = title font size
  show  = to show plot or not
  save  = to save plot or not
  filename : str of image name with img format

  Return
  ------
  plt.show()

  '''  
  _, axs = plt.subplots(nrows, ncols, dpi=dpi, figsize=figsize)
  for i in range(ncols): axs[i].imshow(imgs[i])  
  if titles != []:
    for i in range(ncols): axs[i].title.set_text(titles[i])
    for i in range(ncols): axs[i].title.set_size(title_size)
  for i in range(ncols): axs[i].set_axis_off()
  #for ax in axs:
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    #ax.set_ylim(0,1)
		#ax.minorticks_on()
		#ax.grid()
		#ax.legend()
  plt.tight_layout()
  #---
  if show != True:
    plt.ioff()
  else:
    plt.show;
  #---
  if save == True:
    plt.savefig(filename, transparent=True, dpi=dpi)
    plt.close()
  else: pass







