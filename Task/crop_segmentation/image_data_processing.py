import os

current_dir = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(task_dir)

climatebench_dir = os.path.join(project_dir, "Data", "ClimateBench-M-IMG")
image_chips_folder = os.path.join(project_dir, "Data", "ClimateBench-M-IMG", "image_chips")
crop_classification_split_dir = os.path.join(project_dir, "Data", "ClimateBench-M-IMG", "multi_temporal_crop_classification")

# read training_data.txt and validation_data.txt in crop_classification_split_dir

training_data_file = os.path.join(crop_classification_split_dir, "training_data.txt")
validation_data_file = os.path.join(crop_classification_split_dir, "validation_data.txt")

with open(training_data_file, "r") as f:
    training_data = f.readlines()

with open(validation_data_file, "r") as f:
    validation_data = f.readlines()

# create a folder 'training_chips' in ClimateBench-M-IMG
training_chips_folder = os.path.join(climatebench_dir, "training_chips")
if not os.path.exists(training_chips_folder):
    os.makedirs(training_chips_folder)


for line in training_data:
    line = line.strip()
    merged_img_file = os.path.join(image_chips_folder, line + "_merged.tif")
    mask_img_file = os.path.join(image_chips_folder, line + "_mask.tif")
    # copy the merged image and mask image to training_chips folder
    if os.path.exists(merged_img_file):
        os.system(f"cp {merged_img_file} {training_chips_folder}")
    else:
        print(f"{merged_img_file} does not exist")

validating_chips_folder = os.path.join(climatebench_dir, "validation_chips")
if not os.path.exists(validating_chips_folder):
    os.makedirs(validating_chips_folder)

for line in validation_data:
    line = line.strip()
    merged_img_file = os.path.join(image_chips_folder, line + "_merged.tif")
    mask_img_file = os.path.join(image_chips_folder, line + "_mask.tif")
    # copy the merged image and mask image to training_chips folder
    if os.path.exists(merged_img_file):
        os.system(f"cp {merged_img_file} {validating_chips_folder}")
    else:
        print(f"{merged_img_file} does not exist")
    
