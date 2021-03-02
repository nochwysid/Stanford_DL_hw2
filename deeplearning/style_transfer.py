import numpy as np

import torch
import torch.nn.functional as F


def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    shapes = list(content_current.size())
    loss = 0
    for c in range(shapes[1]):
        for h in range(shapes[2]):
            for w in range(shapes[3]):
                loss += (content_current[:, c, h, w] - content_target[:,c, h, w]) ** 2
    return content_weight * loss        
            
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    shapes = list(features.size())
    N,C,H,W = shapes[0], shapes[1], shapes[2], shapes[3]
    gram = torch.zeros([N, C, C])
    features = features.reshape(N,C, H * W)
    
    for n in range(N):
        gram[n, :] = torch.mm(features[n, :], features[n,:].t())
        

    if normalize:
        return gram / (H * W * C)
    return gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    loss = 0
    for l in range(len(style_layers)):
        gram_feats = gram_matrix(feats[style_layers[l]])
        gram_style = style_targets[l]
        loss += style_weights[l]  * torch.sum((gram_feats - gram_style) ** 2)
        
    return loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    shapes = list(img.size())
    H, W = shapes[2], shapes[3]
    #Thanks Greg for your piazza comment!
    
    #calculate w difference with two "windows", one shifted by 1 relative to the other 
    w_diff_mat = (img[:, :, :, 0:W-1] - img[:, :, :, 1:W]) ** 2
    w_diff = torch.sum(w_diff_mat)
    
    #calculate h difference with two "windows", one shifted by 1 relative to the other
    h_diff_mat = (img[:, :, 0:H-1, :] - img[:, :, 1:H, :]) ** 2
    h_diff = torch.sum(h_diff_mat)
    
    return tv_weight * (w_diff + h_diff)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
