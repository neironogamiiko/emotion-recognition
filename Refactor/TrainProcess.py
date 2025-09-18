import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from pathlib import Path
import logging; import sys
from tqdm.auto import tqdm
from timeit import default_timer as timer

import Utility as ufm
from ModelArchitecture import ClassificationModel
from DataStructure import Data

################
#    LOGGER    #
################

logging.captureWarnings(True)
Path("Logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("Logs/EmotionNNLogs", "w")
    ],
    force=True
)
logger = logging.getLogger(__name__)
sys.excepthook = lambda t, v, tb: logger.error("Uncaught exception", exc_info=(t, v, tb))

####################
#    PARAMETERS    #
####################

DATA_PATH = Path("/home/frasero/.cache/kagglehub/datasets/mstjebashazida/affectnet/versions/1/archive (3)")
TRAIN_PATH = DATA_PATH / "Train"
TEST_PATH = DATA_PATH / "Test"

SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "EmotionNN(state_dict).pth"
FULL_PATH = SAVE_PATH / MODEL_NAME

BATCH_SIZE = 32
SEED = 42
N_CLASSES = 9
LEARNING_RATE = .001
EPOCHS = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Current device: {DEVICE}")

##########################
#    DATA PREPARATION    #
##########################

train_data = Data(path=TRAIN_PATH,
                  augment=True)

test_data = Data(path=TEST_PATH,
                 augment=False)

samples = ufm.find_samples_number(DATA_PATH, logger)
class_counts = [len([file for file in (TRAIN_PATH / n_classes).glob("*.png")]) for n_classes in train_data.classes]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = torch.tensor([class_weights[label] for _, label in train_data.samples])

logger.info(f"train_data.classes: {train_data.classes}")
logger.info(f"first 10 sample labels: {[label for _, label in train_data.samples[:10]]}")


sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)


train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          sampler=sampler)

test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

# ufm.show_random_sample(train_data)
# ufm.show_random_sample(test_data)

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break

# torch.manual_seed(SEED), torch.cuda.manual_seed(SEED)

##################
#    TRAIN NN    #
##################

model = ClassificationModel(N_CLASSES)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

logger.info("\n")

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

#################
#    METRICS    #
#################

#################
#    SAVE NN    #
#################

SAVE_PATH.mkdir(parents=True, exist_ok=True)
checkpoint = {
    "model_state": model.state_dict(),
}
torch.save(checkpoint, FULL_PATH)
logging.info(f"Model and metadata saved to {FULL_PATH}")