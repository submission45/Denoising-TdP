import numpy as np
import torch
import torch.optim as optim

import logging
from tqdm import tqdm
from models.tdp.nn.tdp import TdPModel
from sklearn.metrics import accuracy_score



parameters = {
    "dropout_rate": 0.2,
    "encoder_pool_type": "max",
    "layers": 6,
    "compression": 1,
    "bottleneck": False,
    "pool_steps": [2, 2, 2, 2, 2, 2, 2],
    "activation": {
        "name": "relu",
        "args": { }
        }
    }


def train_model(train_loader, validation_loader, device, directory, label_weights=None, lr=1e-3,
                verbose=True, epochs=200):
    print(epochs)

    val_loss_monitor = np.inf

    model = TdPModel(**parameters, verbose=True, n_classes=2)

    model.to(device)

    if label_weights is not None:
        label_weights = torch.tensor(label_weights).float()
        label_weights = label_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=label_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_values = []
    train_acc_values = []
    eval_loss_values = []
    eval_acc_values = []

    print('\nTraining start')
    for epoch in range(epochs):  # Loop over the dataset.
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        n_update = 0

        bar = tqdm(enumerate(train_loader), ascii=True, ncols=70, total=len(train_loader), colour='green')

        for batch_idx, (data, target) in bar:
            data, target = data.to(device), target
            target = target.type(torch.LongTensor)
            target = target.to(device)
            data = data.unsqueeze(dim=1)

            # Reset the parameter gradients.
            optimizer.zero_grad()

            # Forward + backward + optimize.
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Predict label.
            predicted = torch.argmax(outputs, 1)

            running_loss += loss.item()
            running_accuracy += accuracy_score(
                y_true=target.detach().cpu().numpy(), y_pred=predicted.detach().cpu().numpy()
                )
            n_update += 1

        # Save and print statistics at the end of each training epoch.
        train_loss = running_loss / n_update
        train_acc = running_accuracy / n_update
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        eval_loss, eval_acc = evaluate_during_training(model, criterion, validation_loader, device=device)
        eval_loss_values.append(eval_loss)
        eval_acc_values.append(eval_acc)

        if eval_loss < val_loss_monitor:
            print("Saving model")
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss }, '{}/densenet_epoch_{}.pt'.format(directory, epoch)
                )
            val_loss_monitor = eval_loss

        message = '\n[Epoch {}/{}] Training loss: {:.3f} | Validation loss: {:.3f}'.format(
            epoch+1, epochs, train_loss, eval_loss
            )
        message += '\n[Epoch {}/{}] Training accuracy: {:.2f} % | Validation accuracy: {:.2f} %'.format(
            epoch+1, epochs, train_acc * 100, eval_acc * 100
            )
        if verbose:
            print(message)

        logging.info(message)

    return model


def evaluate_during_training(net, criterion, dataloader, device):
    net.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    n_update = 0

    bar = tqdm(enumerate(dataloader), ascii=True, ncols=70, total=len(dataloader), colour='blue')

    for batch_idx, (data, target) in bar:
        data, target = data.to(device), target

        data = data.unsqueeze(dim=1)
        inputs = data.to(device)
        target = target.type(torch.LongTensor)
        target = target.to(device)

        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, target)
            predicted = torch.argmax(outputs, 1)

            running_loss += loss.item()
            running_accuracy += accuracy_score(
                y_true=target.detach().cpu().numpy(), y_pred=predicted.detach().cpu().numpy()
                )
            n_update += 1
    eval_loss = running_loss / n_update
    eval_acc = running_accuracy / n_update
    return eval_loss, eval_acc

