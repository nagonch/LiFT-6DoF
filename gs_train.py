import torch
import torch.nn as nn
import torch.optim as optim
from dataset import LFDataset
from gsplat.strategy import DefaultStrategy
import viser
from pathlib import Path
import time
from gaussian_splatting.gsplat_viewer import GsplatViewer
from gaussian_splatting.train_tools import (
    get_optimizers_and_params,
    viewer_render_fn as _viewer_render_fn,
    unfreeze_sh,
    batch_rasterize,
    viewer_tick,
    get_scheduler,
    mask_aware_loss,
)
from gaussian_splatting.default_strategy import DefaultStrategy
from random import shuffle
from dataclasses import dataclass
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np


@dataclass
class Config:
    lr: float = 1e-3
    opacity_lr_fraction: float = 0.5
    sh_lr_fraction: float = 0.1
    num_epochs: int = 500
    warmup_epochs: int = 10
    sh_unfreeze_epoch: int = 50
    total_sh_degrees: int = 3
    max_grad_norm: float = 0.1
    scale_bound_min: float = 0.000
    scale_bound_max: float = 1e-2
    enable_viewer: bool = True
    verbose: bool = True
    refine_start_iter: int = 100
    reset_every: int = 2000


def train_splat(
    data,
    config: Config = Config(),
    lr=None,
    n_epochs=None,
    verbose=None,
    enable_viewer=None,
    device="cuda",
):
    if lr is not None:
        config.lr = lr
    if n_epochs is not None:
        config.num_epochs = n_epochs
    if enable_viewer is not None:
        config.enable_viewer = enable_viewer
    if verbose is not None:
        config.verbose = verbose

    @torch.no_grad()
    def viewer_render_fn(camera_state, render_tab_state):
        return _viewer_render_fn(
            camera_state,
            render_tab_state,
            params["means"],
            params["quats"],
            params["scales"],
            torch.sigmoid(params["opacities"]),
            colors=torch.concatenate(
                (params["sh0"], params["sh1"], params["sh2"]), dim=1
            ).unsqueeze(0),
            data_sh_degree=config.total_sh_degrees,
        )

    images = data["images"].to(device)
    poses = data["poses"].to(device)
    masks = data["masks"].to(device)
    camera_matrix = data["camera_matrix"].to(device)
    opacities = data["opacities"].to(device)
    quats = data["quats"].to(device)
    points = nn.Parameter(data["means"].to(device))
    scene_min, scene_max = points.min(0).values, points.max(0).values
    scene_diag = (scene_max - scene_min).norm().clamp(min=1e-6)
    scales = data["scales"].to(device)

    colors = data["colors"].to(device)
    colors_0 = nn.Parameter(colors[:, :1])
    colors_1 = nn.Parameter(colors[:, 1:4])
    colors_2 = nn.Parameter(colors[:, 4:])
    scales = nn.Parameter(scales)
    opacities = nn.Parameter(opacities.to(device))
    quats = nn.Parameter(quats.to(device))
    optimizers = [
        optim.AdamW(
            [
                {"params": params, "lr": lr, "name": name},
            ],
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        for params, lr, name in [
            (points, config.lr, "means"),
            (colors_0, 0, "sh0"),
            (colors_1, 0, "sh1"),
            (colors_2, 0, "sh2"),
            (scales, config.lr, "scales"),
            (opacities, config.lr * config.opacity_lr_fraction, "opacities"),
            (quats, config.lr, "quats"),
        ]
    ]
    params, optimizers_dict = get_optimizers_and_params(optimizers)
    if config.enable_viewer:
        server = viser.ViserServer(verbose=False)
        viewer = GsplatViewer(
            server=server,
            render_fn=viewer_render_fn,
            output_dir=Path("/gs_output"),
            mode="training",
        )
        img_width = images.shape[2]
        img_height = images.shape[1]
        fov = np.arctan2(img_width / 2, camera_matrix[0, 0].item()) * 2
        for i, (pose, image) in enumerate(zip(poses, images)):
            server.scene.add_camera_frustum(
                name=f"cam_pose_{i}",
                aspect=img_width / img_height,
                fov=fov.item(),
                scale=1e-3,
                line_width=0.5,
                image=image.cpu().numpy(),
                wxyz=R.from_matrix(pose[:3, :3].cpu().numpy()).as_quat(
                    scalar_first=True
                ),
                position=pose[:3, 3].cpu().numpy(),
            )
    schedulers = [
        get_scheduler(optimizer, config) for optimizer in optimizers_dict.values()
    ]
    strategy = DefaultStrategy()
    strategy_state = strategy.initialize_state(float(scene_diag))
    strategy.verbose = config.verbose
    strategy.refine_start_iter = config.refine_start_iter
    strategy.reset_every = config.reset_every

    strategy.check_sanity(params, optimizers_dict)

    loss_fn = nn.L1Loss()
    img_indices = list(range(len(images)))
    current_sh_unfreeze_degree = -1
    for epoch in tqdm(range(config.num_epochs), desc="Training Epochs"):
        if (
            epoch % config.sh_unfreeze_epoch == 0
            and epoch != 0
            and current_sh_unfreeze_degree < config.total_sh_degrees
        ):
            current_sh_unfreeze_degree = min(
                config.total_sh_degrees, epoch // config.sh_unfreeze_epoch - 1
            )
            if config.verbose:
                print("Setting current SH degree", current_sh_unfreeze_degree)
            optimizers_dict = unfreeze_sh(
                optimizers_dict, current_sh_unfreeze_degree, config
            )
        shuffle(img_indices)
        if config.enable_viewer:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()
        rendered, alphas, info = batch_rasterize(
            params["means"],
            params["quats"],
            params["scales"],
            torch.sigmoid(params["opacities"]),
            params["sh0"],
            params["sh1"],
            params["sh2"],
            poses,
            camera_matrix,
            images,
            config,
        )
        if config.enable_viewer:
            viewer_tick(viewer, images, epoch, tic)
        strategy.step_pre_backward(params, optimizers_dict, strategy_state, epoch, info)
        loss = mask_aware_loss(loss_fn, rendered, alphas, images, masks)
        loss.backward()
        for optimizer in optimizers_dict.values():
            optimizer.step()
            optimizer.zero_grad()
            clip_grad_norm_(
                parameters=[
                    p
                    for g in optimizer.param_groups
                    for p in g["params"]
                    if p.requires_grad
                ],
                max_norm=config.max_grad_norm,
            )
        for scheduler in schedulers:
            scheduler.step()
        with torch.no_grad():
            quats.data = F.normalize(quats.data, dim=-1)
            scales.data = torch.clamp(
                scales.data,
                min=scene_diag * config.scale_bound_min,
                max=scene_diag * config.scale_bound_max,
            )
        strategy.step_post_backward(
            params, optimizers_dict, strategy_state, epoch, info
        )
        if epoch % 100 == 0 and config.verbose:
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | SH degree: {current_sh_unfreeze_degree}"
            )
    result = {
        "means": params["means"].detach(),
        "colors": torch.concatenate(
            (params["sh0"], params["sh1"], params["sh2"]), dim=1
        ).detach(),
        "scales": params["scales"].detach(),
        "quats": params["quats"].detach(),
        "opacities": torch.sigmoid(params["opacities"]).detach(),
    }
    if config.enable_viewer:
        server.scene.reset()
        server.stop()
    return result


if __name__ == "__main__":
    pass
