import random
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_Building", "BuildingNet")

# Provided COLORS dictionary
COLORS = {
    "moat_ditch": "#ffaa1d", "shutters": "#0dd3ff", "awning": "#ffbbdd", "door": "#7f0000", "wall": "#ff4500",
    "window": "#0000ff", "roof": "#4b008c", "ceiling": "#ff00ff", "floor": "#400000", "chimney": "#999673",
    "stairs": "#60b9bf", "balcony_patio": "#301040", "beam_frame": "#ff4040", "column": "#3d4010",
    "entrance_gate": "#00aaff", "lantern_lamp": "#cc00ff", "garage": "#ff9180", "canopy_gazebo": "#c3e639",
    "gazebo": "#396073", "parapet": "#8a4d99", "parapet_merlon": "#8a4d99", "dormer": "#8c6e69",
    "tower_steeple": "#d6f2b6", "furniture": "#0d2133", "entablature": "#d936b8", "arch": "#b24700",
    "pediment": "#0f7300", "crepidoma": "#bfe1ff", "dome": "#997391", "tympanum": "#e5a173", "buttress": "#40ff73",
    "merlon": "#3662d9", "bridge": "#660036", "ground": "#9370db", "ramp": "#00401a", "fence": "#394173",
    "corridor_path": "#bf3069", "railing_baluster": "#733d00", "plant_tree": "#fa8072", "pond_pool": "#553df2",
    "road": "#ffbfd9", "seats_deck": "#ffaa00", "ground_grass": "#204035", "vehicle": "#396073",
    "cannot_label": "#696969"
}

# JSON data for number to label mapping
json_data = {
    "0": "undetermined", "1": "wall", "2": "window", "3": "vehicle", "4": "roof", "5": "plant", "6": "door",
    "7": "tower", "8": "furniture", "9": "ground", "10": "beam", "11": "stairs", "12": "column", "13": "banister",
    "14": "floor", "15": "chimney", "16": "ceiling", "17": "fence", "18": "pool", "19": "corridor", "20": "balcony",
    "21": "garage", "22": "dome", "23": "road", "24": "gate", "25": "parapet", "26": "buttress", "27": "dormer",
    "28": "lighting", "29": "arch", "30": "awning", "31": "shutters"
}

# Combine the mappings to get number to color mapping
number_to_color = {int(number): COLORS.get(label) for number, label in json_data.items()}

def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Combine the mappings to get number to color mapping
number_to_color = {}
for number, label in json_data.items():
    color = COLORS.get(label)
    if color is None:  # Check if the color is not found in the COLORS dictionary
        color = generate_random_color()  # Generate a random color
    number_to_color[int(number)] = color

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(number_to_color) / 2)
plt.axis('off')

# 각 색상을 사각형으로 표시하고 옆에 번호 표시
for i, (num, color) in enumerate(number_to_color.items()):
    rect = patches.Rectangle((1, i * 0.5), 1, 0.5, linewidth=1, edgecolor='white', facecolor=color)
    ax.add_patch(rect)
    plt.text(2.5, i * 0.5 + 0.25, f"{num}: {color}", verticalalignment='center')

fig.savefig(os.path.join(root_path, "color_mapping.png"))

# Serialize the dictionary to JSON format
json_output = json.dumps(number_to_color, indent=4)

json_path = os.path.join(root_path, "number_color_mapping.json")

# Write to a file
with open(json_path, 'w') as file:
    file.write(json_output)

print("JSON file saved as 'number_to_color_mapping.json'")