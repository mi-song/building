import open3d as o3d
import numpy as np
import os
import json
from collections import Counter

# PLY 파일 및 JSON 파일 경로 설정
file_name = "RESIDENTIALhouse_mesh9936"

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

building_pointcloud = os.path.join(root_path, "POINT_CLOUDS", f"{file_name}.ply")
annotation_json_path = os.path.join(root_path, "point_labels", f"{file_name}_label.json")
color_mapping_path = os.path.join(root_path, "number_color_mapping.json")

with open(color_mapping_path, 'r') as color_file:
    color_mapping = json.load(color_file)

# PLY 파일 불러오기
point_cloud = o3d.io.read_point_cloud(building_pointcloud)

def value_to_color(value):
    # Convert the color hex code to RGB format
    hex_color = color_mapping.get(str(value), "#808080")  # Default grey color
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]

# 복셀 그리드 생성
voxel_size = 0.025
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

# 각 포인트의 복셀 인덱스 계산
points = np.asarray(point_cloud.points)
voxel_indices = np.floor(points / voxel_size).astype(int)

# 인덱스별로 포인트 그룹화 및 원래 포인트 인덱스 저장
point_groups = {}
for idx, (point, voxel_idx) in enumerate(zip(points, voxel_indices)):
    voxel_idx_tuple = tuple(voxel_idx)
    if voxel_idx_tuple not in point_groups:
        point_groups[voxel_idx_tuple] = []
    point_groups[voxel_idx_tuple].append(idx)

# JSON 파일 불러오기 및 파싱
with open(annotation_json_path, 'r') as file:
    annotation_json = json.load(file)

# 각 복셀에 대한 JSON 파일 내의 값 출력
voxel_visuals = []
for voxel_idx, point_indices in point_groups.items():
    values = [annotation_json.get(str(index), None) for index in point_indices]
    most_common_value = Counter(values).most_common(1)[0][0]

    # Determine color based on the most common value using the hash function
    color = value_to_color(most_common_value)
    print(f"{most_common_value}, {color}")

    # Create a voxel as a colored box
    voxel_center = np.array(voxel_idx) * voxel_size + voxel_size / 2
    box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
    box.translate(voxel_center - np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
    box.paint_uniform_color(color)

    voxel_visuals.append(box)

# Combine all boxes into one mesh for efficient visualization
combined_mesh = o3d.geometry.TriangleMesh()
for mesh in voxel_visuals:
    combined_mesh += mesh

# Visualize
o3d.visualization.draw_geometries([combined_mesh])