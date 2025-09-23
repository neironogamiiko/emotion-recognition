import torch
from torch.utils.data import Dataset
import cv2; import random

class Data(Dataset):
    def __init__(self, path, augment = False):
        self.path = path
        self.augment = augment

        self.classes = sorted([class_name.name for class_name in self.path.iterdir() if class_name.is_dir()])
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        self.samples = []
        for sample in self.classes:
            sample_path = self.path / sample
            images = list(sample_path.glob("*.png"))
            self.samples.extend([(img_path, self.class_to_idx[sample]) for img_path in images])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        image_path, label = self.samples[i]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (96, 96))
        image = image.astype('float32') / 255.0

        if self.augment:
            image = self._augment(image)

        image = torch.tensor(image).permute(2, 0, 1)

        return image, label

    def _augment(self, image):
        if random.random() < .5:
            image = cv2.flip(image, 1)

        if random.random() < .5:
            angle = random.uniform(-15,15)
            rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)

        return image