import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W), targets: (N, H, W)

        # Ensure inputs are probabilities by applying softmax
        inputs = F.softmax(inputs, dim=1)

        # Create the one-hot representation of labels
        targets = targets.long()
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # Calculate Focal Loss
        term1 = (1 - inputs) ** self.gamma * torch.log(inputs + 1e-6)  # Add a small value to avoid log(0)
        term2 = inputs ** self.gamma * torch.log(1 - inputs + 1e-6)  # Add a small value to avoid log(0)

        loss = -(targets_one_hot * term1 + (1 - targets_one_hot) * term2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == '__main__':
    # Example usage:
    # pred is the model output with shape (2, 8, 756, 445)
    # label is the ground truth label with shape (2, 756, 445)
    pred = torch.randn(2, 8, 756, 445)
    label = torch.randint(0, 8, (2, 756, 445))  # Random label for demonstration

    # Initialize the Focal Loss module
    focal_loss = FocalLoss()

    # Compute the loss
    loss = focal_loss(pred, label)
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {label.shape}")
    print(f"Focal Loss: {loss.item()}")