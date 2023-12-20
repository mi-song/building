import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def load_point_cloud_and_labels(ply_file, json_file):
    # PLY 파일과 JSON 파일 불러오기
    pcd = o3d.io.read_point_cloud(ply_file)
    with open(json_file, 'r') as file:
        labels = json.load(file)
    return pcd, labels

def assign_colors_to_points(pcd, labels):
    # 색상 맵 생성
    n_colors = len(set(labels.values()))
    cmap = matplotlib.cm.get_cmap('viridis', n_colors)

    # 포인트별 색상 할당
    points = np.asarray(pcd.points)
    colors = [cmap(labels[str(i)])[:3] for i in range(len(points))]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 레이블별 색상 출력
    for label in set(labels.values()):
        color = cmap(label)[:3]
        print(f"레이블 {label}: 색상 RGB = {color}")

    return pcd

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

# 파일 경로 설정
import os

file_name = "COMMERCIALcastle_mesh0365"

building_pointcloud = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "POINT_CLOUDS", f"{file_name}.ply")
annotation_json = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "point_labels", f"{file_name}_label.json")

ply_file = building_pointcloud
json_file = annotation_json

# 데이터 불러오기 및 색상 할당
pcd, labels = load_point_cloud_and_labels(ply_file, json_file)
colored_pcd = assign_colors_to_points(pcd, labels)

# 시각화
visualize_point_cloud(colored_pcd)
