import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import pickle
import json

file_name = "RESIDENTIALhouse_mesh2993.pkl"
root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")
file_path = os.path.join(root_path, "voxel_data_v2", "0.025", file_name)

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

print(max_b)
print(min_b)

b_value = -1

original_points = np.array([[a, c] for (a, b, c), _ in pickle_data if b == b_value])
semantic = np.array([d for (a, b, c), d in pickle_data if b == b_value])

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
points = np.array(x_coords + y_coords)

semantic = [semantic[i] for i in original_indices]

# 각 점을 레이블에 맞는 색상으로 플롯
for i in range(len(points)):
    plt.scatter(points[i, 0], points[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
    plt.text(points[i, 0], points[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')

# for i in range(len(original_points)):
#     plt.scatter(original_points[i, 0], original_points[i, 1], color=color_mapping.get(str(semantic[i]), '#000000'))
#     plt.text(original_points[i, 0], original_points[i, 1], str(semantic[i]), fontsize=8, verticalalignment='bottom')


# 그래프 표시
# plt.show()

# 중심점 계산 함수
def calculate_center(points):
    return points.mean(axis=0)

# 각도 계산 함수
def calculate_angle(point, center):
    return np.arctan2(point[1] - center[1], point[0] - center[0])

# 점들과 색상 정보 결합
combined_points = [(point, color_mapping.get(str(semantic[i]), '#000000')) for i, point in enumerate(points)]

# 중심점 계산
center = calculate_center(points)

# 각도에 따라 점들 정렬
sorted_points = sorted(combined_points, key=lambda x: calculate_angle(x[0], center))

# 마지막 점을 첫 번째 점과 연결
sorted_points.append(sorted_points[0])

# 그래프 그리기
plt.figure()

# 먼저 모든 선(엣지)을 그립니다
for i in range(len(sorted_points)-1):
    point1 = sorted_points[i][0]
    point2 = sorted_points[i+1][0]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='black')  # 엣지를 검은색으로 설정

# 이후 모든 점을 그립니다
for point, color in sorted_points:
    plt.scatter(point[0], point[1], color=color)

plt.scatter(center[0], center[1], color='red', label='Center')  # 중심점 표시
plt.legend()
plt.show()