import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # self.register_buffer("z_vals", torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray))

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = ray_bundle.directions.device)

        # TODO (1.4): Sample points from z values
        sample_points = (ray_bundle.origins.unsqueeze(-1) + z_vals * ray_bundle.directions.unsqueeze(-1)).permute(0,2,1).contiguous()
        ## TODO how about the batch
        sample_lengths=(z_vals * torch.ones(sample_points.shape[:-2], device = ray_bundle.directions.device).unsqueeze(-1)).unsqueeze(-1)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=sample_lengths,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}