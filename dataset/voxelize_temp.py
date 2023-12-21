import open3d as o3d
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt

# 데이터 불러오기
def load_point_cloud_and_labels(ply_file, json_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    with open(json_file, 'r') as file:
        labels = json.load(file)
    return pcd, labels

# 복셀 레이블 할당
def assign_voxel_labels(pcd, labels, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxel_labels = {}

    for i, point in enumerate(np.asarray(pcd.points)):
        voxel_index = tuple(voxel_grid.get_voxel(point))  # 수정된 부분
        voxel_labels.setdefault(voxel_index, []).append(labels[str(i)])

    voxel_majority_labels = {voxel: Counter(lbls).most_common(1)[0][0] for voxel, lbls in voxel_labels.items()}
    return voxel_majority_labels


# 시각화
def visualize_voxel_labels(pcd, voxel_labels, voxel_size, n_colors):
    # 색상 맵 생성
    colors = plt.get_cmap("viridis", n_colors)
    color_map = {label: colors(label)[:3] for label in set(voxel_labels.values())}

    # 복셀 중심에 해당하는 포인트 생성 및 색상 할당
    points = []
    point_colors = []
    for point, label in zip(np.asarray(pcd.points), labels.values()):
        voxel_index = tuple(np.floor(point / voxel_size).astype(int))
        if voxel_index in voxel_labels:
            points.append(point)
            point_colors.append(color_map[voxel_labels[voxel_index]])

    # 색상이 할당된 포인트 클라우드 생성 및 시각화
    voxelized_pcd = o3d.geometry.PointCloud()
    voxelized_pcd.points = o3d.utility.Vector3dVector(points)
    voxelized_pcd.colors = o3d.utility.Vector3dVector(point_colors)

    o3d.visualization.draw_geometries([voxelized_pcd])


import os

file_name = "COMMERCIALcastle_mesh0365"

building_pointcloud = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "POINT_CLOUDS", f"{file_name}.ply")
annotation_json = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "point_labels", f"{file_name}_label.json")

# 데이터 불러오기
pcd, labels = load_point_cloud_and_labels(building_pointcloud, annotation_json)

# 복셀 크기 설정 및 복셀화
voxel_size = 12  # 예시 값, 실제 데이터에 맞게 조정
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

# 복셀 레이블 할당
voxel_labels = assign_voxel_labels(pcd, labels, voxel_size)

# 시각화
n_colors = len(set(labels.values()))
visualize_voxel_labels(pcd, voxel_labels, voxel_size, n_colors)
