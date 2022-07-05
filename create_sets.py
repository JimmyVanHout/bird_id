import os
import random
import shutil
import sys

DIRECTORY_NAMES = ["train", "validation", "test"]
DATA_SPLITS = [0.6, 0.2, 0.2]

def get_split_indices(num_images, data_splits):
    split_indices = []
    for i in range(len(data_splits)):
        indices = None
        if i == 0:
            indices = (0, round(data_splits[i] * num_images) - 1)
        elif i == len(data_splits) - 1:
            indices = (split_indices[i - 1][1] + 1, num_images - 1)
        else:
            indices = (split_indices[i - 1][1] + 1, split_indices[i - 1][1] + round(data_splits[i] * num_images))
        split_indices.append(indices)
    return split_indices

def from_classes_to_sets(directory_names, data_splits):
    for directory_name in directory_names:
        os.mkdir(directory_name)
    count = 0
    for class_name in os.listdir(path="images"):
        images_dir_path = "images/" + class_name
        images_names = os.listdir(path=images_dir_path)
        random.shuffle(images_names)
        split_indices = get_split_indices(len(images_names), data_splits)
        for i in range(len(directory_names)):
            if class_name not in os.listdir(path=directory_names[i]):
                os.mkdir(directory_names[i] + "/" + class_name)
            lower_bound_index = split_indices[i][0]
            upper_bound_index = split_indices[i][1]
            for j in range(lower_bound_index, upper_bound_index + 1):
                shutil.copy2(images_dir_path + "/" + images_names[j], directory_names[i] + "/" + class_name + "/" + images_names[j])
                count += 1
    print("Copied " + str(count) + " images")

if __name__ == "__main__":
    from_classes_to_sets(DIRECTORY_NAMES, DATA_SPLITS)
    sys.exit(0)
