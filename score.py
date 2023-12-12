from sklearn.metrics import roc_auc_score
import numpy as np

def AUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8*len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8*len(saliency_mask))]
    gt = (gt > thres_gt)
    # print("auc")
    # print(gt.sum())
    saliency_mask = (saliency_mask > thres_sal).float()
    return roc_auc_score(gt.numpy(), saliency_mask.numpy())


def sAUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8 * len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8 * len(saliency_mask))]
    gt = (gt > thres_gt).numpy()
    saliency_mask = (saliency_mask > thres_sal).float().numpy()
    # print("sauc")
    # print(gt.sum())
    # print(saliency_mask.sum())
    # 获取正样本和负样本的索引
    positive_indices = np.where(gt > 0)[0]
    negative_indices = np.where(gt == 0)[0]

    # 打乱负样本索引
    np.random.shuffle(negative_indices)

    # 取与正样本相同数量的负样本
    shuffled_indices = np.concatenate([positive_indices, negative_indices[:len(positive_indices)]])

    # 创建新的打乱后的标签
    shuffled_gt = np.zeros_like(gt)
    shuffled_gt[shuffled_indices] = 1

    return roc_auc_score(shuffled_gt, saliency_mask)

def CC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)
    combined = torch.stack((saliency_mask, gt))
    # print("CC: ", torch.corrcoef(combined)[0, 1])
    return float(torch.corrcoef(combined)[0, 1])


import torch

def NSS(saliency_mask, gt):
    """
    计算 Normalized Scanpath Saliency (NSS)
    """
    # 将saliency_mask和gt展平
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)

    # 标准化saliency_mask
    saliency_mask = (saliency_mask - saliency_mask.mean()) / saliency_mask.std()

    # 将gt转换为二进制掩码
    gt = (gt > 0)

    # 计算NSS得分
    nss_score = torch.sum(saliency_mask * gt) / torch.sum(gt)
    # print("NSS: ", nss_score)

    return float(nss_score)

# print(AUC('saliency/maps_1800/000000000071_0.png', 'saliency/image_1800/000000000071_0.png'))