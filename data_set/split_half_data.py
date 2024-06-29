import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder and recreate it
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # Ensure reproducibility
    random.seed(0)

    # Percentage of the dataset to be used
    data_reduction_rate = 0.5
    # Percentage of the reduced dataset to be used as validation set
    split_rate = 0.1

    # Point to the decompressed flower_photos folder
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "flower_data")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # Create folder to save the training set
    train_root = os.path.join(data_root, "train_half")
    mk_file(train_root)
    for cla in flower_class:
        # Create folders for each class
        mk_file(os.path.join(train_root, cla))

    # Create folder to save the validation set
    val_root = os.path.join(data_root, "val_half")
    mk_file(val_root)
    for cla in flower_class:
        # Create folders for each class
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        # Randomly sample images to reduce the dataset by the specified rate
        reduced_images = random.sample(images, k=int(num * data_reduction_rate))
        reduced_num = len(reduced_images)

        # Randomly sample indices for the validation set
        eval_index = random.sample(reduced_images, k=int(reduced_num * split_rate))

        for index, image in enumerate(reduced_images):
            if image in eval_index:
                # Copy files to the validation set
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # Copy files to the training set
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, reduced_num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
