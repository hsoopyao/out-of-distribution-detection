import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from models.classifier import PretrainedClassifier
from utils.dataloader import get_loaders


def compute_msp(model, loader, device):
    model.eval()
    msp_scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            msp_scores.append(torch.max(outputs, dim=1)[0].cpu().numpy())
    return np.concatenate(msp_scores)


def evaluate_ood():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedClassifier().to(device)
    model.load_state_dict(torch.load('cifar10_classifier.pth', map_location=device))
    loaders = get_loaders(batch_size=256)

    # 计算MSP分数
    id_scores = compute_msp(model, loaders['test_id'], device)
    ood_scores = compute_msp(model, loaders['test_ood'], device)

    # 创建标签（ID=1，OOD=0）
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    # 计算AUROC
    auroc = roc_auc_score(labels, scores)
    print(f'AUROC: {auroc:.4f}')
    print(f'ID平均MSP: {np.mean(id_scores):.4f}')
    print(f'OOD平均MSP: {np.mean(ood_scores):.4f}')


if __name__ == '__main__':
    evaluate_ood()