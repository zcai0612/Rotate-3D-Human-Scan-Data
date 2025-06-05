import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
sys.path.append("./thirdparties/econ")

from PIL import Image
from tqdm import tqdm

from kiui.mesh import Mesh 
from utils.render import Renderer
from utils.camera import Camera
from utils.mesh import normalize_vertices, normalize_vertices_with_center_scale
from thirdparties.econ.lib.net.geometry import rot6d_to_rotmat
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, 
)

class ScanProcessor:
    def __init__(self, device='cuda', ortho_views=[0, 45, 90, 180, 270, 315]):
        self.device = device
        self.camera = Camera(device=device)
        self.renderer = Renderer()
        self.ortho_views = ortho_views
        self.smpl_estimator = SMPLEstimator(hps_type="pymafx", device=device)

    def render_front_view(self, mesh, res=512):
        mvps, _, _, _ = self.camera.get_orthogonal_camera(
            views=[0]
        )
        bg_color = self.renderer.get_bg_color('white').to(self.device)
        render_pkgs = self.renderer(mesh, mvp=mvps, h=res, w=res, shading_mode='albedo', bg_color=bg_color)
        
        mesh_rgb = (render_pkgs['image'] * 255).detach().cpu().numpy().astype(np.uint8)
        mesh_rgb_pil = Image.fromarray(mesh_rgb[0]).convert('RGB')
        return mesh_rgb_pil

    def render_ortho_views(self, mesh, res=512):
        mvps, _, _, _ = self.camera.get_orthogonal_camera([-view % 360 for view in self.ortho_views])
        bg_color = self.renderer.get_bg_color('white').to(self.device)
        render_pkgs = self.renderer(mesh, mvp=mvps, h=res, w=res, shading_mode='albedo', bg_color=bg_color)
        mesh_rgb = (render_pkgs['image'] * 255).detach().cpu().numpy().astype(np.uint8)

        mesh_rgb_pil_list = []
        for i in range(len(self.ortho_views)):
            mesh_rgb_pil_list.append(Image.fromarray(mesh_rgb[i]).convert('RGB'))

        return mesh_rgb_pil_list

    def estimate_smpl_dict(self, mesh_rgb_pil):
        mesh_rgb_pil = mesh_rgb_pil.resize((512, 512))
        smpl_dict = self.smpl_estimator.estimate_smpl(mesh_rgb_pil)
        return smpl_dict

    def get_rot_mat(self, smpl_dict):
        orient = smpl_dict["global_orient"].detach()
        rot_mat = rot6d_to_rotmat(orient.view(-1, 6)).view(1, 3, 3)
        return rot_mat

    def extract_y_rotation(self, rot_mat):
        """从完整旋转矩阵中提取仅绕y轴的旋转角度"""
        # 从旋转矩阵提取y轴旋转角度
        # R_y = [[cos(θ), 0, sin(θ)],
        #        [0,      1, 0     ],
        #        [-sin(θ), 0, cos(θ)]]
        sin_y = rot_mat[0, 0, 2]  # R[0,2] = sin(θ)
        cos_y = rot_mat[0, 0, 0]  # R[0,0] = cos(θ)
        
        # 计算y轴旋转角度
        y_angle = torch.atan2(sin_y, cos_y)
        
        # 构建仅绕y轴的旋转矩阵
        cos_theta = torch.cos(y_angle)
        sin_theta = torch.sin(y_angle)
        
        y_rot_mat = torch.tensor([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], device=rot_mat.device, dtype=rot_mat.dtype).unsqueeze(0)
        
        return y_rot_mat

    def rotate_mesh(self, mesh, smpl_mesh, rot_mat):
        # 仅使用绕y轴的旋转分量
        y_rot_mat = self.extract_y_rotation(rot_mat)
        mesh.v = mesh.v @ y_rot_mat.squeeze(0)
        if smpl_mesh is not None:
            smpl_mesh.v = smpl_mesh.v @ y_rot_mat.squeeze(0)
        return mesh, smpl_mesh

    def load_mesh(self, mesh_obj_path, albedo_path=None, smpl_obj_path=None):
        mesh = Mesh.load_obj(mesh_obj_path, albedo_path=albedo_path)
        if smpl_obj_path is not None:
            smpl_mesh = Mesh.load_obj(smpl_obj_path)
            smpl_mesh.v, center, scale = normalize_vertices(smpl_mesh.v, bound=0.9, return_params=True)
            mesh.v = normalize_vertices_with_center_scale(mesh.v, center, scale)
        else:
            smpl_mesh = None
            mesh.v, center, scale = normalize_vertices(mesh.v, bound=1.85/2, return_params=True)
        return mesh, smpl_mesh, center, scale

    def forward(
            self, 
            mesh_obj_path, 
            albedo_path=None, 
            smpl_obj_path=None, 
            render_res=512
        ):
        mesh, smpl_mesh, center, scale = self.load_mesh(mesh_obj_path, albedo_path, smpl_obj_path)
        mesh_rgb_pil = self.render_front_view(mesh, render_res)
        smpl_dict = self.estimate_smpl_dict(mesh_rgb_pil)
        rot_mat = self.get_rot_mat(smpl_dict)
        mesh, smpl_mesh = self.rotate_mesh(mesh, smpl_mesh, rot_mat)

        # front_view_image = self.render_front_view(mesh, render_res)
        # ortho_views_images = self.render_ortho_views(mesh, render_res)
        
        return_dict = {
            "front_view_image_raw": mesh_rgb_pil, # PIL Image_image
            # "front_view_image_processed": front_view_image, # PIL Image
            # "ortho_views_images": ortho_views_images, # list of PIL Image
            "smpl_dict": smpl_dict, # dict
            "rot_mat": rot_mat, # tensor
            "mesh": mesh, # kiui.mesh
            "smpl_mesh": smpl_mesh, # kiui.mesh
        }

        return return_dict        
