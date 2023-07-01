###############################################

import uproot
import numpy as np
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets,models,transforms       
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
cudnn.benchmark = True
plt.ion()

###############################################

## Input data shapes
nStrips = 256
nTimeSlices = 512
nProj = 3
###############################################
###############################################



# ====================================================================== #
# ====================================================================== #
# ====================================================================== #

def parseChargeMaps(fullPath, batch_size=10):

    rootFile =uproot.open(fullPath + ':TPCData/Event')
    tree = rootFile["myChargeArray[3][3][256][512]"]
    num_entries =  len(rootFile["myChargeArray[3][3][256][512]"].array(library="ak"))

    projections = []

    for start_entry in range(0, num_entries, batch_size):
        end_entry = min(start_entry + batch_size, num_entries)

        # Read a batch of events
        batch_projections = tree.array(entry_start=start_entry, entry_stop=end_entry).to_numpy()

        # Process the batch of projections
        processed_batch = batch_projections.astype(float)
        processed_batch = np.sum(processed_batch, axis=2)
        processed_batch = np.moveaxis(processed_batch, 1, -1)
        
        ##threshold
        #processed_batch = (processed_batch > 0.05) * processed_batch

        # Append the processed batch to the projections list
        projections.append(processed_batch)

    # Concatenate the list of projections into a single array
    projections = np.concatenate(projections, axis=0)

    rootFile.close()
    return projections

###############################################
###############################################

def getMergedImages(projections):
    normalized_images = []
    num_events = len(projections)
    for event in range(num_events):

        imageArray = []
        for strip in range(3): 
            imageArray.append(projections[event][:, :, strip])
        color_image = np.stack((imageArray[0], imageArray[1], imageArray[2]), axis=-1)
        max_value = np.max(color_image)
        normalized_images.append( color_image / max_value)
    return normalized_images

###############################################
###############################################



###############################################
###############################################


def getDataML(fullpath):
    projections = parseChargeMaps(fullpath)
    normMergedImage = getMergedImages(projections)
    del projections
    tensor_data = torch.from_numpy(np.array(normMergedImage))
    del normMergedImage
    tensor_data = tensor_data.float()
    padded_tensor_data = padTensorData(tensor_data)
    del tensor_data

    return padded_tensor_data


# ====================================================================== #
# ====================================================================== #
# ====================================================================== #
# ====================================================================== #
# ====================================================================== #


'''
def generator(files, batchSize,filesize):
    for array in uproot.iterate(files, step_size=batchSize, 
                                filter_name=fields, 
                                num_workers = 4, 
                                library="ak"):
        
        features = array["myChargeArray[3][3][256][512]"].to_numpy()
        features = features.astype(float)
        features = np.sum(features, axis=2)
        features = np.moveaxis(features, 1, -1)

        print(len(features))

#       for start_entry in range(0, num_entries, batch_size):


        yield normalized_images


'''


##############################################
##############################################
##############################################

fields = [
    "SimEvent/tracks/tracks.startPos",
    "SimEvent/tracks/tracks.stopPos",
    "Event/myChargeArray*",
]
def padTensorData(batch_data):

    max_dim = max(batch_data.size(1), batch_data.size(2))
    pad_width = max_dim - batch_data.size(2)
    pad_height = max_dim - batch_data.size(1)
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded_tensor = F.pad(batch_data, (0,0,left_pad, right_pad, top_pad, bottom_pad), value=0)
    return padded_tensor


def generator2(files, batchSize):

    for array in uproot.iterate(files, step_size=batchSize,filter_name=fields,  num_workers = 4,  library="ak"):
        features = array["myChargeArray[3][3][256][512]"].to_numpy()
        features = features.astype(float)
        features = np.sum(features, axis=2)
        features = np.moveaxis(features, 1, -1)
        
        # find max pixel in image, from merged image
        merged_image = np.sum(features, axis=(3))

      #  features/=np.amax(merged_image, keepdims=True)
        features /= np.amax(features, axis=(1,2,3), keepdims=True)


        tensor_data = torch.from_numpy(np.array(features))
        tensor_data = tensor_data.float()
        padded_tensor_data = padTensorData(tensor_data)
        #print(padded_tensor_data)

        labels =  np.full((batchSize, ), 1.0)
        yield padded_tensor_data, labels


################################
