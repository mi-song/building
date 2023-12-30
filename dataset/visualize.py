import open3d as o3d
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def visualize_labels_with_colors(labels, cmap):
    # 각 레이블의 색상을 얻기 위한 함수
    def get_label_color(label):
        return cmap(label)[:3]

    # 레이블별 색상 막대 그래프 생성
    unique_labels = sorted(set(labels.values()))
    colors = [get_label_color(label) for label in unique_labels]

    # 막대 그래프 시각화
    plt.figure(figsize=(10, 2))
    plt.bar(unique_labels, [1] * len(unique_labels), color=colors)
    plt.xlabel('label')
    plt.yticks([])
    plt.title('color')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()

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

def load_color_mapping(json_file):
    with open(json_file, 'r') as file:
        color_mapping = json.load(file)
    return color_mapping

def apply_colors_to_point_cloud(pcd, labels, color_mapping):
    points = np.asarray(pcd.points)
    colors = [hex_to_rgb(color_mapping[str(labels[str(i)])]) for i in range(len(points))]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

# 파일 경로 설정
import os

file_name = "COMMERCIALmuseum_mesh1018"

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

building_pointcloud = os.path.join(root_path, "POINT_CLOUDS", f"{file_name}.ply")
annotation_json = os.path.join(root_path, "updated_point_labels", f"{file_name}_label.json")
color_mapping_json = os.path.join(root_path, "number_color_mapping.json")

# 데이터 불러오기
pcd, labels = load_point_cloud_and_labels(building_pointcloud, annotation_json)
color_mapping = load_color_mapping(color_mapping_json)

# 색상 맵 (cmap) 정의
# cmap = matplotlib.cm.get_cmap('viridis', n_colors)

# 색상 적용
colored_pcd = apply_colors_to_point_cloud(pcd, labels, color_mapping)

# 시각화
visualize_point_cloud(colored_pcd)