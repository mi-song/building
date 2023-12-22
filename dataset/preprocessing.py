import open3d as o3d
import numpy as np
import os
import json
from collections import Counter
from tqdm import tqdm
import glob

def find_nearest_points_indices_and_distances(pcd, query_index, k):
    # 포인트 클라우드를 KDTree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # query_index에 해당하는 포인트
    query_point = np.asarray(pcd.points)[query_index]

    # query_point 주변의 k개의 가장 가까운 점과 그 거리 찾기
    [k, idx, sqr_distance] = pcd_tree.search_knn_vector_3d(query_point, k)

    # 제곱 거리를 실제 거리로 변환
    distances = np.sqrt(sqr_distance)
    return idx, distances  # 가장 가까운 점들의 인덱스와 거리 반환

# 포인트 클라우드 데이터 불러오기
root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

# 해당 폴더 내의 모든 .ply 파일 찾기
ply_files = glob.glob(os.path.join(root_path, "POINT_CLOUDS", '*.ply'))

file_names_without_extension = [os.path.splitext(os.path.basename(file))[0] for file in ply_files]

# 파일명만 추출
file_names = [os.path.basename(file) for file in file_names_without_extension]

# file_name = "RESIDENTIALhouse_mesh9936"


for file_name in file_names:
    print(file_name)
    # JSON 파일에서 라벨 데이터 불러오기
    json_file_path = os.path.join(root_path, 'point_labels', f'{file_name}_label.json')
    with open(json_file_path, 'r') as file:
        labels = json.load(file)

    # 라벨이 0인 모든 점들의 인덱스 찾기
    indices_with_label_0 = [int(index) for index, label in labels.items() if label == 0]

    # print("라벨이 0인 모든 점들의 인덱스:", indices_with_label_0)

    building_pointcloud = os.path.join(root_path, "POINT_CLOUDS", f"{file_name}.ply")
    pcd = o3d.io.read_point_cloud(building_pointcloud)

    update_label = {}

    for query_index in tqdm(indices_with_label_0):
        # 가장 가까운 100개의 점과 그 거리 찾기
        nearest_point_indices, nearest_point_distances = find_nearest_points_indices_and_distances(pcd, query_index, 100)

        # 각 인덱스에 해당하는 라벨 찾기
        nearest_point_labels = [labels[str(idx)] for idx in nearest_point_indices]

        # 0을 제외하고 리스트의 요소들을 계산
        filtered_counts = Counter(x for x in nearest_point_labels if x != 0)

        # 가장 흔한 요소 찾기 (최빈값)
        most_common_element = filtered_counts.most_common(1)

        # print("가장 가까운 100개 점의 인덱스:", nearest_point_indices)
        # print("해당 점들까지의 거리:", nearest_point_distances)
        # print("해당 점들의 라벨:", nearest_point_labels)

        if most_common_element:
            labels[str(query_index)] = most_common_element[0][0]
            # print("최빈값 (0 제외):", most_common_element[0][0])
        else:
            labels[str(query_index)] = 0
            # print("0을 제외한 유효한 요소가 없습니다.")

    # print(update_label)

    import json

    # 딕셔너리를 JSON 파일로 저장
    json_save_path = os.path.join(root_path, 'updated_point_labels', f'{file_name}_label.json')

    with open(json_save_path, 'w') as json_file:
        json.dump(labels, json_file, indent=4)
