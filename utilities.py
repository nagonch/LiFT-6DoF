import torch
import viser
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


def clean_point_cloud(points, sigma_max=2):
    z = points[:, 2]
    mean_z = points[:, 2].mean()
    std_z = points[:, 2].std()
    z_max = mean_z + sigma_max * std_z
    valid_mask = z < z_max

    return valid_mask


def backproject_depth_to_pointcloud(
    pixel_indices, depths, camera_matrix, return_scales=False
):
    if pixel_indices is None:
        uu, vv = torch.meshgrid(
            (
                torch.arange(depths.shape[0], device=depths.device),
                torch.arange(depths.shape[1], device=depths.device),
            )
        )
        uu, vv = uu.reshape(-1), vv.reshape(-1)
        pixel_indices = torch.stack((vv, uu), dim=0).T
        depths = depths.reshape(-1)
    inv_camera_matrix = torch.linalg.inv(camera_matrix).double()
    ones = torch.ones(
        (pixel_indices.shape[0], 1),
        device=pixel_indices.device,
        dtype=pixel_indices.dtype,
    ).double()
    uv1 = torch.cat([pixel_indices, ones], dim=1).T  # Shape: [3, N]
    xyz_camera = (inv_camera_matrix @ uv1) * depths
    xyz_camera = xyz_camera.T
    if return_scales:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        scales_x = xyz_camera[:, 2] / fx
        scales_y = xyz_camera[:, 2] / fy
        scales = torch.zeros(xyz_camera.shape[0], 3, device=xyz_camera.device)
        scales[:, 0] = scales_x
        scales[:, 1] = scales_y
        scales[:, 2] = (scales_x + scales_y) / 2
        return xyz_camera, scales
    else:
        return xyz_camera


def create_viser_server() -> viser.ViserServer:
    server = viser.ViserServer(verbose=False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    return server


def run_viser_server(server: viser.ViserServer):
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        server.scene.reset


class Visualizer:
    def __init__(self):
        self.server = create_viser_server()
        self.scene = self.server.scene

    def run(self):
        run_viser_server(self.server)

    def add_point_cloud(self, name, points, colors=None, point_size=1e-4):
        if colors is None:
            colors = np.array([255, 0, 0])
        self.scene.add_point_cloud(name, points, colors=colors, point_size=point_size)

    def add_frame(self, name, frame_T, frames_scale=0.05):
        if not isinstance(frame_T, np.ndarray):
            frame_T = np.array(frame_T)
        position = frame_T[:3, 3]
        rotation = frame_T[:3, :3]
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        self.scene.add_frame(
            name=name,
            position=position,
            wxyz=wxyz,
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
        )

    def add_camera_frustum(self, name, camera_T, camera_matrix, image, scale=0.1):
        image_height, image_width = image.shape[:2]
        if not isinstance(camera_T, np.ndarray):
            camera_T = np.array(camera_T)
        position = camera_T[:3, 3]
        rotation = camera_T[:3, :3]
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        fov = np.arctan2(image_width / 2, camera_matrix[0, 0]) * 2
        self.scene.add_camera_frustum(
            name=name,
            aspect=image_width / image_height,
            fov=fov.item(),
            scale=scale,
            line_width=0.5,
            image=image,
            wxyz=wxyz,
            position=position,
        )


if __name__ == "__main__":
    pass
