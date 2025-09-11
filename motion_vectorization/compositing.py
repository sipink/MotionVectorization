import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib.pyplot as plt
import kornia
from tqdm import tqdm

from .sampling import sampling_layer
from .utils import torch2numpy, init_optimizer, norm_image, compute_image_sdf


def composite_layers(elements, variables, origin, layer_indices, shape, bg_color, 
                     layer_z=None, blur=True, blur_kernel=5, debug=False, device='cuda:0'):
  '''
  Composite layers into a single image.
  Args:
    elements: shape (n_elements, 4, h, w) containing RGBA images for each element.
    variables: list of variables [tx, ty, sx, sy, theta, kx, ky] for each element.
    origin: shape (n_elements, 2) containing origin coordinates for each element.
    layer_indices: dict containing indices of elements to composite for each layer.
    shape: shape of the output image (n, c, h, w).
    bg_color: background color (3,) RGB.
    layer_z: z-order for each element (optional).
    blur: whether to blur the image.
    blur_kernel: kernel size for blurring.
    debug: whether to debug the compositing.
    device: device to use for computation.
  
  Returns:
    composite: shape (n, 3, h, w) containing the composited RGB image.
    alpha: shape (n, 1, h, w) containing the alpha channel.
  '''
  # Composite all layers.
  composite = torch.zeros(shape[0], 3, shape[2], shape[3], dtype=torch.float64, device=device)
  alpha = torch.zeros(shape[0], 1, shape[2], shape[3], dtype=torch.float64, device=device)

  # Add background color.
  composite[:, 0, :, :] = bg_color[2]
  composite[:, 1, :, :] = bg_color[1]
  composite[:, 2, :, :] = bg_color[0]

  elements_warped = sampling_layer(
    elements, variables[0], variables[1], variables[2], variables[3], 
    variables[4], variables[5], variables[6], shape, origin=origin, 
    blur=blur, blur_kernel=blur_kernel, device=device, debug=debug
  )

  elements_warped = torch.clamp(elements_warped, 0, 1)
  if layer_z is not None:
    z_vals = layer_z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, shape[2], shape[3])
    elements_warped[:, 3:4, :, :] = elements_warped[:, 3:4, :, :] * z_vals
  else:
    z_vals = torch.ones(shape[0], 1, shape[2], shape[3], dtype=torch.float64, device=device)

  for layer_idx in layer_indices:
    for element_idx in layer_indices[layer_idx]:
      element_alpha = elements_warped[element_idx:element_idx+1, 3:4, :, :]
      element_rgb = elements_warped[element_idx:element_idx+1, :3, :, :]
      composite = composite * (1 - element_alpha) + element_rgb * element_alpha
      alpha = alpha * (1 - element_alpha) + element_alpha

  return composite, alpha


def loss_fn(pred, targets, variables, layer_z=None, default_variables=None, 
            loss_type='mse', single_scale=False, use_mask=False, p_weight=0.01, device='cuda:0'):
  '''
  Compute loss for the composited image.
  '''
  pred_rgb = pred[:, :3, :, :]
  pred_alpha = pred[:, 3:4, :, :]
  
  if single_scale:
    targets = [targets[0]]
  
  rgb_loss = torch.tensor(0.0, device=device)
  rgb_scales_loss = torch.tensor(0.0, device=device)
  alpha_loss = torch.tensor(0.0, device=device)
  alpha_scales_loss = torch.tensor(0.0, device=device)
  
  for i, target in enumerate(targets):
    if i == 0:
      scale_factor = 1.0
    else:
      scale_factor = 2.0 ** i
    
    # Downsample prediction to match target resolution
    if i > 0:
      pred_rgb_scaled = F.interpolate(pred_rgb, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
      pred_alpha_scaled = F.interpolate(pred_alpha, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    else:
      pred_rgb_scaled = pred_rgb
      pred_alpha_scaled = pred_alpha
    
    target_rgb = target[:, :3, :, :]
    target_alpha = target[:, 3:4, :, :] if target.shape[1] > 3 else torch.ones_like(pred_alpha_scaled)
    
    if use_mask:
      mask = target_alpha > 0.5
      if loss_type == 'mse':
        rgb_loss_i = torch.mean((pred_rgb_scaled - target_rgb) ** 2 * mask)
        alpha_loss_i = torch.mean((pred_alpha_scaled - target_alpha) ** 2)
      else:  # l1
        rgb_loss_i = torch.mean(torch.abs(pred_rgb_scaled - target_rgb) * mask)
        alpha_loss_i = torch.mean(torch.abs(pred_alpha_scaled - target_alpha))
    else:
      if loss_type == 'mse':
        rgb_loss_i = torch.mean((pred_rgb_scaled - target_rgb) ** 2)
        alpha_loss_i = torch.mean((pred_alpha_scaled - target_alpha) ** 2)
      else:  # l1
        rgb_loss_i = torch.mean(torch.abs(pred_rgb_scaled - target_rgb))
        alpha_loss_i = torch.mean(torch.abs(pred_alpha_scaled - target_alpha))
    
    if i == 0:
      rgb_loss += rgb_loss_i
      alpha_loss += alpha_loss_i
    else:
      rgb_scales_loss += rgb_loss_i * (0.5 ** i)
      alpha_scales_loss += alpha_loss_i * (0.5 ** i)
  
  # Parameter regularization loss
  params_loss = torch.tensor(0.0, device=device)
  if default_variables is not None and p_weight > 0:
    for var, default_var in zip(variables, default_variables):
      params_loss += torch.mean((var - default_var) ** 2)
    params_loss *= p_weight
  
  return rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss


def sdf_loss_fn(pred_sdf, target_sdf):
  '''
  Compute SDF loss between predicted and target SDF.
  '''
  return torch.mean((pred_sdf - target_sdf) ** 2)


def compute_sdf(mask):
  '''
  Compute signed distance field from binary mask.
  '''
  return compute_image_sdf(mask)


def dsample(img, factor=2):
  '''
  Downsample image by a factor.
  '''
  b, c, h, w = img.shape
  img = F.interpolate(img, size=(h//factor, w//factor), mode='bilinear', align_corners=False)
  return img


def place_shape(shapes, xs, ys, sxs, sys, thetas, kxs, kys, width, height, keep_alpha=False):
  '''
  Place shapes at specified locations with transformations.
  '''
  frame = np.zeros((height, width, 4 if keep_alpha else 3), dtype=np.float32)
  
  for shape, x, y, sx, sy, theta, kx, ky in zip(shapes, xs, ys, sxs, sys, thetas, kxs, kys):
    h, w = shape.shape[:2]
    
    # Create transformation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Scale and rotate
    M = np.array([[sx * cos_theta, -sx * sin_theta, x - w*sx/2],
                  [sy * sin_theta, sy * cos_theta, y - h*sy/2]], dtype=np.float32)
    
    # Apply transformation
    transformed_shape = cv2.warpAffine(shape, M, (width, height), flags=cv2.INTER_LINEAR)
    
    if keep_alpha:
      # Alpha compositing
      alpha = transformed_shape[:, :, 3:4] if transformed_shape.shape[2] > 3 else np.ones_like(transformed_shape[:, :, :1])
      frame[:, :, :3] = frame[:, :, :3] * (1 - alpha) + transformed_shape[:, :, :3] * alpha
      frame[:, :, 3:4] = frame[:, :, 3:4] * (1 - alpha) + alpha
    else:
      # Simple overlay
      mask = transformed_shape.sum(axis=2) > 0
      frame[mask] = transformed_shape[mask, :3]
  
  return frame * 255


def get_distance_to_region(alpha, res=10):
  '''
  Compute distance to region for each pixel.
  '''
  region = alpha.clone()
  region[region < 0.5] = -1
  region[region >= 0.5] = 1
  region = kornia.filters.gaussian_blur2d(region, (5, 5), (5, 5))

  # Uniformly sample pixels outside of region.
  b, c, h, w = alpha.shape
  cn, rn = w // res, h // res
  xs = torch.linspace(-1, 1, cn, dtype=torch.float64)
  ys = torch.linspace(-1, 1, rn, dtype=torch.float64)
  grid_x, grid_y = torch.meshgrid(ys, xs)
  grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0)
  
  # Get the closest distance to region.
  sc = torch.tensor([[h, w]])
  region_s = F.grid_sample(region, grid, padding_mode='reflection')
  region_mask = 100 * (torch.sigmoid(100 * (torch.abs(region_s) - 0.99))).view(-1, 1)
  xs_ = torch.linspace(0, 1, cn, dtype=torch.float64)
  ys_ = torch.linspace(0, 1, rn, dtype=torch.float64)
  coords = torch.stack(torch.meshgrid(ys_, xs_), dim=-1).view(-1, 2)
  dists = torch.cdist(coords, coords) + region_mask.T
  min_dists, _ = torch.min(dists, dim=1)
  min_dists = min_dists.view(-1, c, rn, cn)

  # Resample into full image.
  full_xs = torch.linspace(-1, 1, w, dtype=torch.float64)
  full_ys = torch.linspace(-1, 1, h, dtype=torch.float64)
  full_grid_x, full_grid_y = torch.meshgrid(full_ys, full_xs)
  full_grid = torch.stack([full_grid_y, full_grid_x], dim=-1).unsqueeze(0)
  full_res = F.grid_sample(min_dists, full_grid, padding_mode='reflection')
  full_res = full_res * torch.sign(region)
  return full_res


if __name__ == '__main__':
  pass