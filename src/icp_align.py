import open3d as o3d
import numpy as np


def align_colored_point_clouds_np(source_np, target_np, voxel_size=0.02):
    def np_to_pcd(np_array):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_array[:, :3])
        colors = np_array[:, 3:6]
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def pcd_to_np(pcd):
        return np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors)))

    def preprocess(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        return pcd_down, fpfh

    def estimate_normals(pcd, voxel_size):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

    def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    ):
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 2.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    voxel_size * 2.5
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000),
        )

    def refine_with_colored_icp(source, target, init_trans, voxel_size):
        return o3d.pipelines.registration.registration_colored_icp(
            source,
            target,
            voxel_size,
            init_trans,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
            ),
        )

    source_pcd = np_to_pcd(source_np)
    target_pcd = np_to_pcd(target_np)

    estimate_normals(source_pcd, voxel_size)
    estimate_normals(target_pcd, voxel_size)

    source_down, source_fpfh = preprocess(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess(target_pcd, voxel_size)

    global_result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )

    icp_result = refine_with_colored_icp(
        source_pcd, target_pcd, global_result.transformation, voxel_size
    )

    source_pcd.transform(icp_result.transformation)

    return icp_result.transformation, pcd_to_np(source_pcd)


if __name__ == "__main__":
    pass
