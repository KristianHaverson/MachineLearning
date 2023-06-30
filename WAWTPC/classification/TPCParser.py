###############################################

import uproot
import numpy as np
import torch.nn.functional as F

###############################################

## Input data shapes
nStrips = 256
nTimeSlices = 512
nProj = 3
#projections = np.zeros((nStrips,nTimeSlices, nProj))
normalized_images = np.zeros((nStrips,nTimeSlices))
###############################################
###############################################


def parseChargeMaps(fullPath, batch_size=100):

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






def padTensorData(batch_data):
    max_dim = max(batch_data.size(1), batch_data.size(2))
    pad_width = max_dim - batch_data.size(2)
    pad_height = max_dim - batch_data.size(1)
    
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    print(top_pad)
    print(bottom_pad)
    print(left_pad)
    print(right_pad)

    padded_tensor = F.pad(batch_data, (0,0,left_pad, right_pad, top_pad, bottom_pad), value=0)
    

    return padded_tensor



###############################################
###############################################










