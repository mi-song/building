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

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

# 파일 경로 설정
import os

file_name = "COMMERCIALcastle_mesh0365"

building_pointcloud = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "POINT_CLOUDS", f"{file_name}.ply")
annotation_json = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet", "point_labels", f"{file_name}_label.json")

pcd, labels = load_point_cloud_and_labels(building_pointcloud, annotation_json)
n_colors = len(set(labels.values()))

# 색상 맵 (cmap) 정의
cmap = matplotlib.cm.get_cmap('viridis', n_colors)

# 색상이 할당된 포인트 클라우드 생성
colored_pcd = assign_colors_to_points(pcd, labels)

# 레이블 별 색상 시각화
visualize_labels_with_colors(labels, cmap)

# 포인트 클라우드 시각화
visualize_point_cloud(colored_pcd)