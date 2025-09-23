from torch.utils.data import DataLoader, Dataset
import cv2; import numpy as np
from pathlib import Path
import random

class Data(Dataset):
    def __init__(self, path:Path, split:str):
        """
        :param path: path to Dataset, that contains Train and Test data.
        """
        # <-----    PATH PARAMETERS    -----> #
        self.input_path = path
        self.split = split # "Train" / "Test"
        self.output_path = Path("/home/frasero/PycharmProjects/Datasets") / split
        self.output_path.mkdir(parents=True, exist_ok=True)
        # <---------------------------------> #

        self.image_size = (96, 96)
        self.data = []
        self.class_to_idx = {}

    def _augmentation(self, image):
        augment = random.choice(["flip", "rotate", "brightness"])
        if augment == "flip":
            image = cv2.flip(image, flipCode=1)

        elif augment == "rotate":
            angle = random.choice([30, 45, 90])
            (heigh, weight) = image.shape[:2]
            rotation = cv2.getRotationMatrix2D((weight//2, heigh//2), angle, scale=1.)
            image = cv2.warpAffine(image, rotation, (heigh, weight))

        elif augment == "brightness":
            factor = random.uniform(.7, 1.3)
            image = np.clip(image*factor, 0, 255).astype(np.uint8)

        return image

    def _prepare_data(self, image):
        classes = sorted([directory.name for directory in self.input_path.iterdir() if directory.is_dir()])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        class_counts = {} # dictionary {'class_name': num_samples}

        for class_name in classes:
            class_directory = self.input_path / class_name
            save_class_directory = self.output_path / class_name
            save_class_directory.mkdir(parents=True, exist_ok=True)

            counter = 0

            for image_path in class_directory.glob("*"):
                image = cv2.imread(str(image_path))
                if image is None: continue

                image = cv2.resize(image, self.image_size)
                image_save_path = save_class_directory / f"{counter:05d}.jpg"
                cv2.imwrite(str(image_save_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                self.data.append((image, self.class_to_idx[class_name]))

                counter += 1

            class_counts[class_name] = counter


