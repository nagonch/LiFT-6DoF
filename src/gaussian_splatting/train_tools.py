import torch
from gsplat import rasterization
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def get_optimizers_and_params(optimizers_list):
    optimizers_dict = {}
    parameter_dict = {}
    for optimizer in optimizers_list:
        for group in optimizer.param_groups:
            optimizers_dict[group.get("name")] = optimizer
            parameter_dict[group.get("name")] = group["params"][0]
    parameter_dict = torch.nn.ParameterDict(parameter_dict).cuda()
    return parameter_dict, optimizers_dict


def mask_aware_loss(loss_fn, pred_color, pred_alpha, target, mask, weights=None):
    if weights is None:
        weights = torch.ones_like(mask)
    inter = (pred_alpha.squeeze(-1) * mask).sum()
    union = pred_alpha.squeeze(-1).sum() + mask.sum() + 1e-6
    loss_outer = 1.0 - 2.0 * inter / union
    loss_total = (
        0.5
        * (loss_fn(pred_color[mask > 0], target[mask > 0]) * weights[mask > 0]).mean()
        + 0.5 * loss_outer
    )
    return loss_total


def mask_aware_loss_img(loss_fn, pred_color, pred_alpha, target, mask):
    inter = pred_alpha.squeeze(-1) * mask
    union = pred_alpha.squeeze(-1) + mask + 1e-6
    loss_outer = 1.0 - 2.0 * inter / union
    loss_inner = torch.abs(
        pred_color * (mask > 0).float()[..., None]
        - target * (mask > 0).float()[..., None]
    ).sum(dim=-1)
    loss_total = 0.5 * loss_inner + 0.5 * loss_outer
    return loss_total


def get_scheduler(optimizer, config):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_epochs - config.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )
    return scheduler


def viewer_tick(viewer, images, epoch, tic):
    viewer.lock.release()
    num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
    num_train_rays_per_step = images.shape[1] * images.shape[2]
    num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
    viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
    viewer.update(epoch, num_train_rays_per_step)


def batch_rasterize(
    points,
    quats,
    scales,
    opacities,
    colors_0,
    colors_1,
    colors_2,
    poses,
    camera_matrix,
    images,
    config,
    render_mode="RGB",
    backgrounds=None,
    enable_sh=True,
):
    if enable_sh:
        colors = torch.concatenate((colors_0, colors_1, colors_2), dim=1).unsqueeze(0)
    else:
        colors = torch.concatenate(
            (colors_0, torch.zeros_like(colors_1), torch.zeros_like(colors_2)), dim=1
        ).unsqueeze(0)

    render, alphas, info = rasterization(
        means=points.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors,
        viewmats=torch.linalg.inv(poses).unsqueeze(0),
        Ks=torch.stack(
            [
                camera_matrix,
            ]
            * poses.shape[0]
        ).unsqueeze(0),
        width=images.shape[2],
        height=images.shape[1],
        sh_degree=config.total_sh_degrees,
        packed=False,
        render_mode=render_mode,
        backgrounds=backgrounds,
    )
    return render[0], alphas[0], info


def unfreeze_sh(optimizers, degree, config):
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            if group.get("name") == f"sh{degree}":
                group["lr"] = config.sh_lr_fraction * config.lr
    return optimizers


def viewer_render_fn(
    camera_state,
    render_tab_state,
    points,
    quats,
    scales,
    opacities,
    colors,
    data_sh_degree=3,
):
    if render_tab_state.preview_render:
        width = render_tab_state.render_width
        height = render_tab_state.render_height
    else:
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K((width, height))
    c2w = torch.from_numpy(c2w).float().cuda()
    K = torch.from_numpy(K).float().cuda()
    viewmat = c2w.inverse()

    RENDER_MODE_MAP = {
        "rgb": "RGB",
        "depth(accumulated)": "D",
        "depth(expected)": "ED",
        "alpha": "RGB",
    }
    render_tab_state.backgrounds = (255.0, 255.0, 255.0)  # white background
    render_colors, render_alphas, info = rasterization(
        points,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, S, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=(
            min(render_tab_state.max_sh_degree, data_sh_degree)
            if data_sh_degree is not None
            else None
        ),
        near_plane=render_tab_state.near_plane,
        far_plane=render_tab_state.far_plane,
        radius_clip=render_tab_state.radius_clip,
        eps2d=render_tab_state.eps2d,
        backgrounds=torch.tensor([render_tab_state.backgrounds]).cuda() / 255.0,
        render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
        rasterize_mode=render_tab_state.rasterize_mode,
        camera_model=render_tab_state.camera_model,
        packed=False,
    )
    render_tab_state.total_gs_count = len(points)
    render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
    render_colors = render_colors[0, ..., 0:3]
    render_colors = torch.clip(render_colors, min=0, max=1)
    renders = render_colors.cpu().numpy()
    return renders


if __name__ == "__main__":
    pass
