import os
import random
import shutil


def split_and_copy_dataset(source_image_folder, source_label_folder, train_image_folder, train_label_folder, test_image_folder, test_label_folder, split_ratio):
    """Split the image and labels from a dataset into another dataset path

    Args:
        source_image_folder (str): Where is located the images
        source_label_folder (str): Where is licated the labels
        train_image_folder (str): Where the images for training go
        train_label_folder (str): Where the labels for training go
        test_image_folder (_type_): Where the images for validation go
        test_label_folder (_type_): Where the images for validation go
        split_ratio (_type_): ...
    """
    image_files = os.listdir(source_image_folder)
    label_files = os.listdir(source_label_folder)

    random.shuffle(image_files)
    random.shuffle(label_files)

    split_index = int(len(image_files) * split_ratio)

    # Copy training data
    for i in range(split_index):
        image_src = os.path.join(source_image_folder, image_files[i])
        label_src = os.path.join(source_label_folder, label_files[i])
        shutil.copy(image_src, train_image_folder)
        shutil.copy(label_src, train_label_folder)

    # Copy testing data
    for i in range(split_index, len(image_files)):
        image_src = os.path.join(source_image_folder, image_files[i])
        label_src = os.path.join(source_label_folder, label_files[i])
        shutil.copy(image_src, test_image_folder)
        shutil.copy(label_src, test_label_folder)

# Define your dataset folders and split ratio
datasets = [
    {
        'source_image_folder': '/content/YOLO-QR-datasets/Dataset 1/images',
        'source_label_folder': '/content/YOLO-QR-datasets/Dataset 1/labels',
    },
    # {
    #     'source_image_folder': '/content/YOLO-QR-datasets/Dataset 2/images',
    #     'source_label_folder': '/content/YOLO-QR-datasets/Dataset 2/labels',
    # }
]

train_image_folder = '/content/data/images/train'
train_label_folder = '/content/data/labels/train'
test_image_folder = '/content/data/images/val'
test_label_folder = '/content/data/labels/val'

split_ratio = 0.9

# Split and copy each dataset
for dataset in datasets:
    split_and_copy_dataset(
        dataset['source_image_folder'],
        dataset['source_label_folder'],
        train_image_folder,
        train_label_folder,
        test_image_folder,
        test_label_folder,
        split_ratio
    )


