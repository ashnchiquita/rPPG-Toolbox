import os
import yaml
import csv

def replace_in_csv(file_path, old_str, new_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [ [cell.replace(old_str, new_str) for cell in row] for row in reader ]
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def replace_in_yaml(file_path, old_str, new_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    def recursive_replace(obj):
        if isinstance(obj, dict):
            return {k: recursive_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_replace(i) for i in obj]
        elif isinstance(obj, str):
            return obj.replace(old_str, new_str)
        else:
            return obj

    data = recursive_replace(data)

    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(data, file)

def traverse_and_replace(folder_path, old_str, new_str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                replace_in_csv(os.path.join(root, file), old_str, new_str)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                replace_in_yaml(os.path.join(root, file), old_str, new_str)

# Example usage:
d_ubfc = '/mnt/d/dataset/UBFC-rPPG/cache'
c_pure = '/mnt/c/dataset/PURE/cache'
f_ubfc = '/mnt/f/chiquita/dataset/UBFC-rPPG/cache/'
e_pure = '/mnt/e/chiquita/dataset/PURE/cache/'
traverse_and_replace(c_pure, '/mnt/d', '/mnt/c')
