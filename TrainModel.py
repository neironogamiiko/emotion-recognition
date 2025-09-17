from pathlib import Path
import logging; import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import cv2; import random
import UtilityForModel as ufm
from tqdm.auto import tqdm

logging.captureWarnings(True)
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/EmotionDetectorLogs", "w")
    ],
    force=True
)
logger = logging.getLogger(__name__)
sys.excepthook = lambda t, v, tb: logger.error("Uncaught exception", exc_info=(t, v, tb))

DATA_PATH = Path("/home/frasero/.cache/kagglehub/datasets/mstjebashazida/affectnet/versions/1/archive (3)")
TRAIN_PATH = DATA_PATH / "Train"
TEST_PATH = DATA_PATH / "Test"

SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "FaceRecognition(state_dict).pth"
FULL_PATH = SAVE_PATH / MODEL_NAME

BATCH_SIZE = 32
SEED = 42
N_CLASSES = 9
LEARNING_RATE = .001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Current device: {DEVICE}")

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

samples = ufm.find_samples_number(DATA_PATH, logger)

train_data = Data(path=TRAIN_PATH,
                  augment=True)

test_data = Data(path=TEST_PATH,
                 augment=False)

class_counts = [len([file for file in (TRAIN_PATH / n_classes).glob("*.png")]) for n_classes in train_data.classes]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = torch.tensor([class_weights[label] for _, label in train_data.samples])

sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          sampler=sampler)

test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

ufm.show_random_sample(train_data)
ufm.show_random_sample(test_data)
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, N_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

torch.manual_seed(SEED), torch.cuda.manual_seed(SEED)

model = ClassificationModel()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

logger.info("\n")

from timeit import default_timer as timer
train_time_start = timer()

for epoch in tqdm(range(EPOCHS)):
    logger.info(f"Epoch: {epoch}")
    ufm.train_step(model=model,
                   data_loader=train_loader,
                   criterion=criterion,
                   optimizer=optimizer,
                   accuracy=ufm.accuracy,
                   device=DEVICE,
                   logger=logger)

    predictions = ufm.test_step(model=model,
                                data_loader=test_loader,
                                criterion=criterion,
                                accuracy=ufm.accuracy,
                                device=DEVICE,
                                logger=logger)

    test_accuracy = ufm.accuracy(torch.tensor([label for _, label in test_data.samples]).to(DEVICE), predictions)
    scheduler.step(test_accuracy)

logger.info("\n")

train_time_end = timer()
total_time = ufm.print_train_time(start=train_time_start,
                                  end=train_time_end,
                                  device=DEVICE,
                                  logger=logger)


SAVE_PATH.mkdir(parents=True, exist_ok=True)
checkpoint = {
    "model_state": model.state_dict(),
}
torch.save(checkpoint, FULL_PATH)
logging.info(f"Model and metadata saved to {FULL_PATH}")