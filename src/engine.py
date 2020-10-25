import torch
import torch.nn as nn
import numpy as np

# Class inspired by Abhishek Thakur's book:
# "Approaching (Almost) Any Machine Learning
# Problem"
class Engine:
    @staticmethod
    def train(data_loader, model, optimizer, device, scheduler=None):
        """Function for training the model for one epoch
        :param data_loader: torch data_loader
        :param model: model
        :param optimizer: torch optimizer
        :param device: device
        :param scheduler: learning scheduler
        """
        # setting the model to the training mode
        model.train()
        model.to(device)

        # list to track training loss
        training_loss = list()

        for data in data_loader:
            inputs = data["image"].to(device)
            labels = data["target"].to(device)

            # Clearing the gradients
            # This approach is faset than optimizer.zero_grad()
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            for param in model.parameters():
                param.grad = None

            outputs = model(inputs)
            # Calculating the loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            print(loss.item())

            training_loss.append(loss.item())
            # Compute the grad
            loss.backward()
            if scheduler is not None:
                scheduler.step(loss)

            optimizer.step()
        return training_loss

    @staticmethod
    def evaluate(data_loader, model, device):
        # initialize empy lists to store predictions
        # and targets
        final_predictions = []
        final_targets = []

        # putting the model to eval mode
        model.eval()
        model.to(device)
        with torch.no_grad():
            for data in data_loader:
                inputs = data["image"].to(device)
                labels = data["target"].to(device)

                # making predictions
                predictions = model(inputs)

                predictions = predictions.cpu().numpy().tolist()
                targets = data["target"].cpu().numpy().tolist()
                final_predictions.extend(predictions)
                final_targets.extend(targets)

        return final_predictions, final_targets
