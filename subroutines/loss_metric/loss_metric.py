"""
The following code was produced for the Journal paper 
"Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning"
by D. Dais, İ. E. Bal, E. Smyrou, and V. Sarhosis published in "Automation in Construction"
in order to apply Deep Learning and Computer Vision with Python for crack detection on masonry surfaces.

In case you use or find interesting our work please cite the following Journal publication:

D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces 
using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. 
https://doi.org/10.1016/j.autcon.2021.103606.

@article{Dais2021,
author = {Dais, Dimitris and Bal, İhsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},
doi = {10.1016/j.autcon.2021.103606},
journal = {Automation in Construction},
pages = {103606},
title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},
url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},
volume = {125},
year = {2021}
}

The paper can be downloaded from the following links:
https://doi.org/10.1016/j.autcon.2021.103606
https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning/stats

The code used for the publication can be found in the GitHb Repository:
https://github.com/dimitrisdais/crack_detection_CNN_masonry

Author and Moderator of the Repository: Dimitris Dais

For further information please follow me in the below links
LinkedIn: https://www.linkedin.com/in/dimitris-dais/
Email: d.dais@pl.hanze.nl
ResearchGate: https://www.researchgate.net/profile/Dimitris_Dais2
Research Group Page: https://www.linkedin.com/company/earthquake-resistant-structures-promising-groningen
YouTube Channel: https://www.youtube.com/channel/UCuSdAarhISVQzV2GhxaErsg  

Your feedback is welcome. Feel free to reach out to explore any options for collaboration.
"""

#%%

import numpy as np
import cv2


#%%
# Define metrics to be used for evaluation of the trained model using NumPy instead of tensorflow tensors
#

def DilateMask(mask, threshold=0.5, iterations=1):
    """
    receives mask and returns dilated mask
    """

    kernel = np.ones((5,5),np.uint8)
    mask_dilated = mask.copy()
    mask_dilated = cv2.dilate(mask_dilated,kernel,iterations = iterations)
    # Binarize mask after dilation
    mask_dilated = np.where(mask_dilated>threshold, 1., 0.)  

    return mask_dilated
	
def Recall_np(y_true, y_pred, threshold=0.5):
    
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(),half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    possible_positives = y_true_f.sum()
    recall = (true_positives + eps) / (possible_positives + eps)
    return recall

def Precision_np(y_true, y_pred, threshold=0.5):
    
    eps = 1e-07
    y_true_f = y_true.flatten().astype('float32')
    half = (np.ones(y_true_f.shape)*threshold).astype('float32')
    y_pred_f = np.greater(y_pred.flatten(),half).astype('float32')
    true_positives = (y_true_f * y_pred_f).sum()
    predicted_positives = y_pred_f.sum()
    precision = (true_positives + eps) / (predicted_positives + eps)
    
    return precision

def F1_score_np(recall,precision):
   
    eps = 1e-07
    return 2*((precision*recall)/(precision+recall+eps))
