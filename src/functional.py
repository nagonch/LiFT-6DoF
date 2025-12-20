import torch
from .utilities import backproject_depth_to_pointcloud, clean_point_cloud
from .spherical_harmonics import get_SF_coeffs
from .disparity import get_frame_disparity


def cluster_rays(LF, depth_center, masks, camera_matrix, baseline):
    s_size, t_size, u_size, v_size = masks.shape
    uv_inds_middle = torch.stack(
        torch.where(masks[s_size // 2, t_size // 2] > 0), dim=1
    )
    n_points = uv_inds_middle.shape[0]
    cluster_ids = torch.arange(n_points, device=LF.device)

    disparities_uv = (camera_matrix[0, 0] * baseline) / (
        depth_center[uv_inds_middle[:, 0], uv_inds_middle[:, 1]] + 1e-8
    )

    ss, tt = torch.meshgrid(
        torch.arange(s_size, device=LF.device),
        torch.arange(t_size, device=LF.device),
        indexing="ij",
    )
    st_inds = torch.stack((ss, tt), dim=-1).reshape(-1, 2)

    disparities_uv = disparities_uv.tile(s_size * t_size)
    cluster_ids = cluster_ids.tile(s_size * t_size)
    st_inds = st_inds.repeat(n_points, 1)
    uv_inds_middle = uv_inds_middle.repeat(s_size * t_size, 1)

    uv_inds_st = (
        uv_inds_middle
        + (torch.tensor([s_size // 2, t_size // 2]).cuda() - st_inds)
        * disparities_uv[..., None]
    )
    uv_inds_st = torch.round(uv_inds_st).to(torch.int32)
    in_bounds = (
        (uv_inds_st[:, 0] >= 0)
        & (uv_inds_st[:, 1] >= 0)
        & (uv_inds_st[:, 0] < u_size)
        & (uv_inds_st[:, 1] < v_size)
    )

    weights = torch.zeros(
        uv_inds_st.shape[0], dtype=torch.bool, device=uv_inds_st.device
    )
    valid_uv = uv_inds_st[in_bounds]
    valid_st = st_inds[in_bounds]
    weights[in_bounds] = masks[
        valid_st[:, 0], valid_st[:, 1], valid_uv[:, 0], valid_uv[:, 1]
    ].bool()
    uv_inds_st[:, 0] = torch.clip(uv_inds_st[:, 0], 0, u_size - 1)
    uv_inds_st[:, 1] = torch.clip(uv_inds_st[:, 1], 1, v_size - 1)

    depths = (camera_matrix[0, 0] * baseline) / (disparities_uv + 1e-8)
    angle_phi = torch.arctan2((st_inds[:, 1] - t_size // 2) * baseline, depths)
    angle_theta = torch.arctan2((st_inds[:, 0] - s_size // 2) * baseline, depths)
    angles = torch.stack((angle_phi, angle_theta), axis=-1)
    point_clouds, scales = backproject_depth_to_pointcloud(
        uv_inds_middle[:, [1, 0]], depths, camera_matrix, return_scales=True
    )
    del depths
    del angle_phi, angle_theta
    del disparities_uv
    sorting_order = torch.argsort(cluster_ids)
    st_inds = st_inds[sorting_order]
    uv_inds_st = uv_inds_st[sorting_order]
    point_clouds = point_clouds[sorting_order]
    angles = angles[sorting_order]
    weights = weights[sorting_order]
    scales = scales[sorting_order]
    colors = LF[
        st_inds[:, 0], st_inds[:, 1], uv_inds_st[:, 0], uv_inds_st[:, 1]
    ].reshape(n_points, -1, 3)
    angles = angles.reshape(n_points, -1, 2)
    point_clouds = point_clouds.reshape(n_points, -1, 3)[:, 0, :]
    scales = scales.reshape(n_points, -1, 3).mean(axis=1)
    weights = weights.reshape(n_points, -1)
    valid_mask = clean_point_cloud(point_clouds)
    return (
        colors[valid_mask],
        angles[valid_mask],
        point_clouds[valid_mask],
        weights[valid_mask],
        scales[valid_mask],
    )


def get_gs_initialization(frame):
    camera_matrix = frame["camera_matrix"]
    baseline = frame["baseline"]
    LF = frame["LF"]
    masks = frame["masks"]
    disparity = get_frame_disparity(frame)
    depth = camera_matrix[0, 0] * baseline / (disparity + 1e-8)
    mask_middle = masks[masks.shape[0] // 2, masks.shape[1] // 2]
    point_clouds = backproject_depth_to_pointcloud(None, depth, camera_matrix)
    point_clouds = point_clouds[mask_middle.reshape(-1) > 0]
    LF *= masks[..., None]

    colors, angles, means, weights, scales = cluster_rays(
        LF, depth, masks, camera_matrix, baseline
    )
    sh_coeffs = get_SF_coeffs(angles, colors, weights)
    return point_clouds, means, sh_coeffs, scales


if __name__ == "__main__":
    pass
