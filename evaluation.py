from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from model import FakeDetectorModel
from config import Config


def evaluate(model: FakeDetectorModel, test_data: Dataset, config: Config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.BCELoss()
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size)

    if use_cuda:
        model.cuda()
        criterion.cuda()

    model.eval()
    test_losses = 0
    test_correct = 0
    predictions = []

    with torch.no_grad():
        for test_input, test_label, test_mask in tqdm(test_dataloader):
            test_label = test_label.to(device)
            test_input = test_input.to(device)
            test_mask = test_mask.to(device)

            output = model(test_input, test_mask)
            output = torch.squeeze(output)
            batch_loss = criterion(output, test_label.float())
            test_losses += batch_loss.item()
            prediction = torch.round(output)
            test_correct += (prediction == test_label).sum().item()
            predictions.extend(prediction.tolist())

    test_accuracy = test_correct / len(test_dataloader.dataset)

    print("Test loss: {:.3f}".format(test_losses))
    print("Test accuracy: {:.3f}".format(test_accuracy))

    return test_losses, test_accuracy, predictions
