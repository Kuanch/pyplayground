import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net import network
from dataset import dataloader
from utils.utils import plot_classes_preds


def train():
    model = network.get_network('resnet18', num_classes=10).cuda()
    dataset = dataloader.TorchLoader('CIFAR10', train_batch_size=64, test_batch_size=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    writer = SummaryWriter()

    for epoch in range(10):
        running_loss = 0.
        for i in range(len(dataset.train_loader)):
            inputs, labels = dataset.read_train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, i)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            if i % 700 == 699:
                print('draw training predictions')
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(model, inputs, labels),
                                  global_step=epoch)

            if i % 700 == 699:
                with torch.no_grad():
                    TP = FP = TN = FN = 0
                    for j in range(10):
                        images, labels = dataset.read_test()
                        output = model(images)
                        _, class_preds_batch = torch.max(output, 1)
                        np_labels = labels.cpu().numpy()
                        np_preds = class_preds_batch.cpu().numpy()

                        TP += sum(np_labels[np.where(np_preds == 1)[0]] == 1)
                        FP += sum(np_labels[np.where(np_preds == 1)[0]] == 0)
                        TN += sum(np_labels[np.where(np_preds == 0)[0]] == 0)
                        FN += sum(np_labels[np.where(np_preds == 0)[0]] == 1)

                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    writer.add_scalar("Val/precision", precision, epoch)
                    writer.add_scalar("Val/recall", recall, epoch)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train()