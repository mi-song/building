import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import pickle
import json
from tqdm import tqdm

# file_name = "RESIDENTIALhouse_mesh2993.pkl"

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

folder_path = os.path.join(root_path, "updated_point_labels")
file_names = os.listdir(folder_path)

file_names = [name.replace('_label.json', '') for name in file_names]

def sort_ccw(points):
    # 시작점을 y축 기준으로 가장 작은 값, 그리고 x축 기준으로 가장 작은 값이 되도록 선택
    start_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    starting_point = points[start_idx]

    # 중심점 계산
    center = np.mean(points, axis=0)

    # 각도를 기준으로 점들을 정렬
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]

    # 기존 정렬 과정 유지
    reordered_points_list = [starting_point.tolist()]
    for _ in range(1, len(sorted_points)):
        last_point = np.array(reordered_points_list[-1])
        remaining_points = np.array([p.tolist() for p in sorted_points if p.tolist() not in reordered_points_list])

        # 가까운 점들 찾기
        nearest_remaining_points = []
        for remaining_point in remaining_points:
            if remaining_point[0] == last_point[0] - 1 or remaining_point[0] == last_point[0] + 1 or remaining_point[1] == last_point[1] - 1 or remaining_point[1] == last_point[1] + 1:
                nearest_remaining_points.append(remaining_point.tolist())  # NumPy 배열을 리스트로 변환

        # x 또는 y축의 방향성을 고려한 점 찾기
        direction_consistent_points = []
        for point in nearest_remaining_points:
            for next_point in remaining_points:
                if np.array_equal(next_point, point):
                    continue
                # x 또는 y축의 방향성 확인
                if (last_point[0] == point[0] and point[0] == next_point[0]) or \
                   (last_point[1] == point[1] and point[1] == next_point[1]):
                    direction_consistent_points.append(point)
                    break

        # 거리 계산
        if len(nearest_remaining_points) > 0:
            distances = np.linalg.norm(np.array(nearest_remaining_points) - last_point, axis=1)
            min_distance = np.min(distances)
            closest_points = [p for p, d in zip(nearest_remaining_points, distances) if d == min_distance]

            if direction_consistent_points:
                # 방향성을 고려한 점 중 하나 선택
                closest_idx = nearest_remaining_points.index(direction_consistent_points[0])
            elif len(closest_points) > 1:
                # 세 점이 일직선상에 있는지 확인
                inline_points = []
                for point in closest_points:
                    for next_point in remaining_points:
                        if np.array_equal(next_point, point):
                            continue
                        # 기울기 계산
                        gradient1 = (point[1] - last_point[1]) / (point[0] - last_point[0]) if point[0] != last_point[0] else float('inf')
                        gradient2 = (next_point[1] - point[1]) / (next_point[0] - point[0]) if next_point[0] != point[0] else float('inf')

                        # 기울기가 같으면 일직선상에 있는 것으로 간주
                        if gradient1 == gradient2:
                            inline_points.append(point)
                            break

                # 일직선상에 있는 점이 있는 경우
                if inline_points:
                    closest_idx = nearest_remaining_points.index(inline_points[0])
                else:
                    # 거리가 가장 짧은 점 선택
                    closest_idx = np.argmin(distances)
            else:
                closest_idx = np.argmin(distances)

            reordered_points_list.append(nearest_remaining_points[closest_idx])

    reordered_points = np.array(reordered_points_list)

    # 정렬된 점들을 플롯
    plt.scatter(reordered_points[:, 0], reordered_points[:, 1])
    plt.scatter(starting_point[0], starting_point[1], color='red')  # 시작점

    # 각 점에 순서를 표시
    for idx, point in enumerate(reordered_points):
        plt.text(point[0], point[1], str(idx), fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # 플롯 표시
    plt.show()

    return reordered_points


for file_name in tqdm(file_names):
    # print(file_name)

    file_path = os.path.join(root_path, "voxel_data_v2", "0.025", f"{file_name}.pkl")

    if not os.path.exists(file_path):
        continue

    if file_name in ["PUBLICmonastery_mesh0235", "COMMERCIALmuseum_mesh1018", "RESIDENTIALvilla_mesh0183"]:
        continue

    # 파일을 바이너리 읽기 모드로 열기
    with open(file_path, 'rb') as file:
        # pickle 객체 로드
        pickle_data = pickle.load(file)

    color_mapping_path = os.path.join(root_path, "number_color_mapping.json")

    with open(color_mapping_path, 'r') as color_file:
        color_mapping = json.load(color_file)

    b_values = [b for (a, b, c), d in pickle_data]

    # 최대값과 최소값 계산
    max_b = max(b_values)
    min_b = min(b_values)

    whole_points = []

    for b_value in range(min_b, max_b+1):
        # b_value = -1

        original_points = np.array([[a, c] for (a, b, c), _ in pickle_data if b == b_value])
        semantic = np.array([d for (a, b, c), d in pickle_data if b == b_value])

        # 제외할 semantic 레이블
        exclude_labels = [5, 9, 14, 17, 18, 23]

        # 필터링
        filtered_points = original_points[~np.isin(semantic, exclude_labels)]
        filtered_semantic = semantic[~np.isin(semantic, exclude_labels)]

        original_points = filtered_points
        semantic = filtered_semantic

        # x축이 동일한 것들에 대한 y축 최대, 최소값

        unique_x = np.unique(original_points[:, 0])

        x_max_min = {x: (original_points[original_points[:, 0] == x, 1].max(), original_points[original_points[:, 0] == x, 1].min()) for x in unique_x}

        # y축이 동일한 것들에 대한 x축 최대, 최소값
        unique_y = np.unique(original_points[:, 1])
        y_max_min = {y: (original_points[original_points[:, 1] == y, 0].max(), original_points[original_points[:, 1] == y, 0].min()) for y in unique_y}

        # Create a dictionary to track original indices
        original_indices = []

        x_coords = []
        for x, (y_max, y_min) in x_max_min.items():
            max_idx = np.where((original_points[:, 0] == x) & (original_points[:, 1] == y_max))[0][0]
            min_idx = np.where((original_points[:, 0] == x) & (original_points[:, 1] == y_min))[0][0]

            if y_max != y_min:
                x_coords.append([x, y_max])
                original_indices.append(max_idx)
                x_coords.append([x, y_min])
                original_indices.append(min_idx)
            else:
                x_coords.append([x, y_max])
                original_indices.append(max_idx)

        # y축이 동일한 것들에 대한 x축 최대, 최소값 좌표 및 인덱스 구하기
        y_coords = []
        z_coords = []
        for y, (x_max, x_min) in y_max_min.items():
            max_idx = np.where((original_points[:, 1] == y) & (original_points[:, 0] == x_max))[0][0]
            min_idx = np.where((original_points[:, 1] == y) & (original_points[:, 0] == x_min))[0][0]

            if x_max != x_min:
                y_coords.append([x_max, y])
                original_indices.append(max_idx)
                y_coords.append([x_min, y])
                original_indices.append(min_idx)
            else:
                y_coords.append([x_max, y])
                original_indices.append(max_idx)

        # Combine the coordinates
        x_y_coords_arr = np.array(x_coords + y_coords)

        # print(x_y_coords_arr)

        sorted_x_y_coords_arr = sort_ccw(x_y_coords_arr)

        x_y_coords = [list(row) for row in x_y_coords_arr]
        z_coords = [b_value] * len(x_y_coords)

        points = [sublist + [z_coords[i]] for i, sublist in enumerate(x_y_coords)]

        semantic = [semantic[i] for i in original_indices]

        # # 각 점을 레이블에 맞는 색상으로 플롯
        # for i in range(len(x_y_coords_arr)):
        #     plt.scatter(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
        #     plt.text(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')

        # for i in range(len(original_points)):
        #     plt.scatter(original_points[i, 0], original_points[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
        #     plt.text(original_points[i, 0], original_points[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')

        # 그래프 표시
        # plt.show()

        coords_semantic = [[sublist, semantic[i]] for i, sublist in enumerate(points)]

        whole_points.extend(coords_semantic)

    coords_semantic_folder_path = os.path.join(root_path, "pointernet", "dataset", "coords_semantic")
    coords_semantic_file_path = os.path.join(coords_semantic_folder_path, f"{file_name}.pickle")

    with open(coords_semantic_file_path, 'wb') as file:
        pickle.dump(whole_points, file)