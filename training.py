from tqdm import tqdm
import torch
from torch import device
from torch.utils.data import Dataset, DataLoader

from model import FakeDetectorModel
from config import Config


def train(model: FakeDetectorModel, train_data: Dataset, val_data: Dataset, config: Config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size)

    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_val_losses = []
    epoch_val_accuracies = []

    if use_cuda:
        model.cuda()
        criterion.cuda()

    for epoch in range(config.epochs):
        model.train()
        print("----------------------------")
        print("Epoch ", epoch + 1)
        train_losses = 0
        train_correct = 0
        for train_input, train_label, train_mask in tqdm(train_dataloader):
            train_label = train_label.to(device)
            train_input = train_input.to(device)
            train_mask = train_mask.to(device)

            output = model(train_input, train_mask)
            output = torch.squeeze(output)
            batch_loss = criterion(output, train_label.float())

            train_losses += batch_loss.item()
            train_correct += (torch.round(output) == train_label).sum().item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # validate
        val_losses, val_correct = validate(model, device, val_dataloader, criterion)

        train_accuracy = train_correct / len(train_dataloader.dataset)
        val_accuracy = val_correct / len(val_dataloader.dataset)

        print("Train loss: {:.3f}".format(train_losses))
        print("Train accuracy: {:.3f}".format(train_accuracy))
        print("Val loss: {:.3f}".format(val_losses))
        print("Val accuracy: {:.3f}".format(val_accuracy))

        epoch_train_losses.append(train_losses)
        epoch_train_accuracies.append(train_accuracy)
        epoch_val_losses.append(val_losses)
        epoch_val_accuracies.append(val_accuracy)

        torch.save({
            'epoch': (epoch + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'train_accuracy': train_accuracy,
            'val_loss': val_losses,
            'val_accuracy': val_accuracy,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size
        }, "./models/model_{}.pt".format((epoch+1)))

    return epoch_train_losses, epoch_train_accuracies, epoch_val_losses, epoch_val_accuracies


def validate(model: FakeDetectorModel, device: device, val_dataloader: DataLoader, criterion):
    model.eval()
    val_losses = 0
    val_correct = 0

    with torch.no_grad():
        for val_input, val_label, val_mask in val_dataloader:
            val_label = val_label.to(device)
            val_input = val_input.to(device)
            val_mask = val_mask.to(device)

            output = model(val_input, val_mask)
            output = torch.squeeze(output)
            batch_loss = criterion(output, val_label.float())
            val_losses += batch_loss.item()
            val_correct += (torch.round(output) == val_label).sum().item()

    return val_losses, val_correct
