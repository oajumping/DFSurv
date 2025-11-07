import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(batch_size=32):
    X_train = torch.randn(100, 128)
    y_train = torch.randint(0, 2, (100,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    X_test = torch.randn(30, 128)
    y_test = torch.randint(0, 2, (30,))
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader
