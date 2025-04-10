import torch
import torch.nn as nn
from models.classifier import PretrainedClassifier
from utils.dataloader import get_loaders


def train_model(epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("正在初始化数据加载器...")
    loaders = get_loaders()
    print(f"训练集批次: {len(loaders['train'])}")

    model = PretrainedClassifier().to(device)
    print("模型结构:")
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\n----- Epoch {epoch + 1}/{epochs} -----")
        model.train()
        for batch, (inputs, labels) in enumerate(loaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            print(f"\n开始验证，共有{len(loaders['test_id'])}个批次")
            for batch_idx, (inputs, labels) in enumerate(loaders['test_id']):
                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loaders['test_id']):
                    print(f"验证进度: {batch_idx + 1}/{len(loaders['test_id'])}")

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{epochs} | Test Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'cifar10_classifier.pth')
    print("模型已保存为 'cifar10_classifier.pth'")


if __name__ == '__main__':
    train_model()