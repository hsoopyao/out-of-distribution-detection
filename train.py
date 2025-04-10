import torch
import torch.nn as nn
from models.classifier import PretrainedClassifier
from utils.dataloader import get_loaders


def train_model(epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedClassifier().to(device)

    # 冻结部分层（可选）
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    # for param in model.base_model.fc.parameters():
    #     param.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    loaders = get_loaders()

    for epoch in range(epochs):
        model.train()
        for batch, (inputs, labels) in enumerate(loaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loaders['test_id']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{epochs} | Test Acc: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'cifar10_classifier.pth')


if __name__ == '__main__':
    train_model()