import open3d as o3d
import numpy as np
import os
import json
from collections import Counter
import pickle
import glob
from tqdm import tqdm

# PLY 파일 및 JSON 파일 경로 설정
file_name = "RESIDENTIALhouse_mesh2993"

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

# 해당 폴더 내의 모든 .ply 파일 찾기
ply_files = glob.glob(os.path.join(root_path, "POINT_CLOUDS", '*.ply'))

file_names_without_extension = [os.path.splitext(os.path.basename(file))[0] for file in ply_files]

# 파일명만 추출
file_names = [os.path.basename(file) for file in file_names_without_extension]

for file_name in tqdm(file_names):
    if file_name == "RESIDENTIALhouse_mesh9936":
    # print(file_name)
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
        voxel_data = []

        for voxel_idx, point_indices in point_groups.items():
            # 각 포인트 인덱스에 대응하는 레이블 값들을 가져옴
            values = [annotation_json.get(str(index), None) for index in point_indices]

            print(values)

            # 0이 아닌 값들만 추출
            non_zero_values = [value for value in values if value != 0]

            # 0이 아닌 값이 있으면 그 중 가장 흔한 값을 찾음, 아니면 most_common_value를 0으로 설정
            if non_zero_values:
                most_common_value = Counter(non_zero_values).most_common(1)[0][0]
            else:
                most_common_value = 0

            # Determine color based on the most common value using the hash function
            color = value_to_color(most_common_value)

            # Create a voxel as a colored box
            voxel_center = np.array(voxel_idx) * voxel_size + voxel_size / 2

            box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
            box.translate(voxel_center - np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
            box.paint_uniform_color(color)

            voxel_visuals.append(box)

            voxel_data.append([voxel_idx, most_common_value])

        pickle_folder_path = os.path.join(root_path, 'voxel_data', str(voxel_size))

        if not os.path.exists(pickle_folder_path):
            os.makedirs(pickle_folder_path)

        # pickle_file_path = os.path.join(pickle_folder_path, f"{file_name}.pkl")  # 저장할 파일 경로
        # with open(pickle_file_path, 'wb') as file:
        #     pickle.dump(voxel_data, file)

        # Combine all boxes into one mesh for efficient visualization
        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in voxel_visuals:
            combined_mesh += mesh

        # Visualize
        o3d.visualization.draw_geometries([combined_mesh])

        # 복셀 중심의 y값 계산
        y_values = [voxel_center[1] for voxel_center in point_groups.keys()]

        # 최대 및 최소 y값 찾기
        min_y, max_y = min(y_values), max(y_values)
        # print(f"최소 y값: {min_y}, 최대 z값: {max_y}")

# # 특정 z값 범위에 해당하는 복셀만 시각화
# z_min_threshold = ...  # 최소 z값 설정
# z_max_threshold = ...  # 최대 z값 설정
# filtered_voxel_visuals = []
# for voxel_center, box in zip(point_groups.keys(), voxel_visuals):
#     if z_min_threshold <= voxel_center[2] * voxel_size <= z_max_threshold:
#         filtered_voxel_visuals.append(box)
#
# # 필터링된 복셀 시각화
# o3d.visualization.draw_geometries(filtered_voxel_visuals)
