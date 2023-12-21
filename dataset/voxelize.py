import open3d as o3d
import numpy as np
import os
import json

# PLY 파일 및 JSON 파일 경로 설정
file_name = "COMMERCIALcastle_mesh0365"
building_pointcloud = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "POINT_CLOUDS", f"{file_name}.ply")
annotation_json_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "point_labels", f"{file_name}_label.json")

# PLY 파일 불러오기
point_cloud = o3d.io.read_point_cloud(building_pointcloud)

# 복셀 그리드 생성
voxel_size = 0.05
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
for voxel_idx, point_indices in point_groups.items():
    # point_indices가 리스트라면, 각 인덱스에 대한 값을 찾아 출력
    values = [annotation_json.get(str(index), None) for index in point_indices]
    print(f"Voxel {voxel_idx} contains values: {values}")
