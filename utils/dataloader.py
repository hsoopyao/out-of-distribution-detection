from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size=128):
    # 使用ImageNet标准化参数
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 保持SVHN使用相同标准化参数
    test_ood_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_id = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_ood = datasets.SVHN(root='./data', split='test', download=True, transform=test_ood_transform)

    return {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        'test_id': DataLoader(test_id, batch_size=batch_size, num_workers=4),
        'test_ood': DataLoader(test_ood, batch_size=batch_size, num_workers=4)
    }