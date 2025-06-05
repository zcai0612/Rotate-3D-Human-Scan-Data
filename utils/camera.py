import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import math 
import random

from smplx.lbs import batch_rodrigues

class Camera:
    def __init__(self, device='cuda'):
        self.device = device

    def get_orthogonal_camera(self, views):
        """Initialize orthogonal cameras for rendering."""
        rot = batch_rodrigues(
            torch.tensor([
                [0, np.radians(angle), 0] for angle in views 
            ])).float().to(self.device) 
        extrinsic = torch.eye(4)[None].expand(len(views), -1, -1).clone() .to(self.device)
        extrinsic[:, :3, :3] = rot
        intrinsic = torch.eye(4)[None].expand(len(views), -1, -1).clone() .to(self.device)
        R = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3).float()   
        intrinsic[:, :3, :3] = R  
        mvps = torch.bmm(intrinsic, extrinsic) 
        extrinsic = extrinsic

        return mvps, rot, extrinsic, intrinsic