
import os
import random
import shutil

# Define paths
base_dir = '/home/ladmin/harish/pothole.v18i.yolov5pytorch'
train_img_dir = os.path.join(base_dir, 'train/images')
train_lbl_dir = os.path.join(base_dir, 'train/labels')
valid_img_dir = os.path.join(base_dir, 'valid/images')
valid_lbl_dir = os.path.join(base_dir, 'valid/labels')
test_img_dir = os.path.join(base_dir, 'test/images')
test_lbl_dir = os.path.join(base_dir, 'test/labels')

# Get all image files
all_images = os.listdir(train_img_dir)
random.shuffle(all_images)

# Define split sizes
val_size = 100
test_size = 100
train_size = len(all_images) - val_size - test_size

# Split files
val_files = all_images[:val_size]
test_files = all_images[val_size:val_size + test_size]

def move_files(files, dest_img_dir, dest_lbl_dir):
    for f in files:
        # Move image
        shutil.move(os.path.join(train_img_dir, f), os.path.join(dest_img_dir, f))
        # Move label
        label_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(os.path.join(train_lbl_dir, label_file), os.path.join(dest_lbl_dir, label_file))

# Move validation files
move_files(val_files, valid_img_dir, valid_lbl_dir)
print(f'Moved {len(val_files)} files to validation set.')

# Move test files
move_files(test_files, test_img_dir, test_lbl_dir)
print(f'Moved {len(test_files)} files to test set.')

print('Dataset splitting complete.')
