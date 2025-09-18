import random
import cv2
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
import logging

def find_samples_number(path, logger):
    train_test_data = [file.name for file in path.iterdir()]
    min_samples = {}
    for dataset in train_test_data:
        class_path = path / dataset
        classes = [class_name.name for class_name in class_path.iterdir() if class_name.is_dir()]
        img_counts = {cls: len(list((class_path / cls).iterdir())) for cls in classes}
        logger.info(f"{dataset} samples per class: {img_counts}")
        min_samples[dataset] = min(img_counts.values())

    return min_samples

def show_random_sample(dataset):
    i = random.randint(0, len(dataset) - 1)
    image, label = dataset[i]

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype("uint8")  # back to 0–255

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    class_name = dataset.classes[label]
    cv2.imshow(f"Label: {class_name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def accuracy(y_true, y_predicted):
    correct = torch.eq(y_true, y_predicted).sum().item()
    accuracy = (correct / len(y_predicted)) * 100
    return accuracy

def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               accuracy,
               device: torch.device,
               logger: logging.Logger):

    model.to(device)
    train_loss, train_accuracy = 0, 0
    for X,y in data_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        train_loss += loss
        train_accuracy += accuracy(y, logits.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    logger.info(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.2f}%")

def test_step(model: nn.Module,
              data_loader: DataLoader,
              criterion: nn.Module,
              accuracy,
              device: torch.device,
              logger: logging.Logger):

    test_loss, test_accuracy = 0, 0
    all_predictions = []
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_predictions = logits.argmax(dim=1)

            loss = criterion(logits, y)
            test_loss += loss
            test_accuracy += accuracy(y, test_predictions)

            all_predictions.append(test_predictions)

        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)

        predictions = torch.cat(all_predictions, dim=0)
        logger.info(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")

    return predictions

def print_train_time(start: float,
                     end: float,
                     device: torch.device,
                     logger: logging.Logger):

    total_time = end - start
    logger.info(f"Train time on {device}: {total_time:.3f} seconds")

    return total_time