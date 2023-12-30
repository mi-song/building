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

    # # 정렬된 점들을 플롯
    # plt.scatter(reordered_points[:, 0], reordered_points[:, 1])
    # plt.scatter(starting_point[0], starting_point[1], color='red')  # 시작점
    #
    # # 각 점에 순서를 표시
    # for idx, point in enumerate(reordered_points):
    #     plt.text(point[0], point[1], str(idx), fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # 플롯 표시
    # plt.show()

    reordered_indices = [original_indices[np.where((points == point).all(axis=1))[0][0]] for point in reordered_points]

    # 변경된 순서와 인덱스 배열 반환
    return reordered_points, reordered_indices

    # return reordered_points

# 좌표를 합치는 함수
def merge_coordinates(coords, lbls):
    merged_coords = []
    merged_labels = []

    # 좌표 배열이 비어있는지 확인
    if len(coords) == 0:
        return np.array(merged_coords), np.array(merged_labels)

    # 첫 번째와 마지막 레이블이 같은 경우, 첫 번째 레이블의 좌표들을 마지막에 추가
    if lbls[0] == lbls[-1]:
        coords = np.append(coords, [coords[0]], axis=0)
        lbls = lbls + [lbls[0]]

    current_group = [coords[0]]
    current_label = lbls[0]

    # 연속적인 좌표 그룹을 찾기 위한 반복문
    for coord, label in zip(coords[1:], lbls[1:]):
        if label == current_label:
            # 같은 레이블의 좌표를 현재 그룹에 추가
            current_group.append(coord)
        else:
            # 새로운 레이블이 나타날 때, 이전 그룹의 중간 지점 계산 및 저장
            avg_coord = np.mean(current_group, axis=0)
            merged_coords.append(avg_coord)
            merged_labels.append(current_label)

            # 새로운 그룹 시작
            current_group = [coord]
            current_label = label

    # 마지막 그룹 처리 (첫 번째와 마지막이 같은 경우 이미 추가되었으므로 생략)
    if lbls[0] != lbls[-1] and current_group:
        avg_coord = np.mean(current_group, axis=0)
        merged_coords.append(avg_coord)
        merged_labels.append(current_label)

    return np.array(merged_coords), np.array(merged_labels)



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

        sorted_x_y_coords_arr, reordered_indices = sort_ccw(x_y_coords_arr)

        x_y_coords_arr = sorted_x_y_coords_arr

        x_y_coords = [list(row) for row in x_y_coords_arr]
        z_coords = [b_value] * len(x_y_coords)

        points = [sublist + [z_coords[i]] for i, sublist in enumerate(x_y_coords)]

        # semantic = [semantic[i] for i in original_indices]

        semantic = [semantic[i] for i in reordered_indices]

        semantic = np.array(semantic)

        # 좌표 합치기
        merged_coordinates_fixed, merged_labels_fixed = merge_coordinates(points, semantic)

        # 합쳐진 좌표와 레이블을 플롯
        plt.scatter(merged_coordinates_fixed[:, 0], merged_coordinates_fixed[:, 1], c=merged_labels_fixed, marker='x')
        plt.colorbar(label='Merged Semantic Labels')
        plt.title('Scatter Plot of Merged Coordinates with Labels')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()


        # 각 점을 레이블에 맞는 색상으로 플롯
        # for i in range(len(x_y_coords_arr)):
        #     plt.scatter(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
        #     plt.text(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')

        for i in range(len(x_y_coords_arr)):
            plt.scatter(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1],
                        color=color_mapping.get(str(semantic[i]), '#000000'))
            # 인덱스를 표시
            plt.text(x_y_coords_arr[i, 0], x_y_coords_arr[i, 1], str(i), fontsize=8, verticalalignment='bottom')

        # for i in range(len(original_points)):
        #     plt.scatter(original_points[i, 0], original_points[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
        #     plt.text(original_points[i, 0], original_points[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')

        # 그래프 표시
        plt.show()

        coords_semantic = [[sublist, semantic[i]] for i, sublist in enumerate(points)]

        whole_points.extend(coords_semantic)

    coords_semantic_folder_path = os.path.join(root_path, "pointernet", "dataset", "coords_semantic")
    coords_semantic_file_path = os.path.join(coords_semantic_folder_path, f"{file_name}.pickle")

    with open(coords_semantic_file_path, 'wb') as file:
        pickle.dump(whole_points, file)