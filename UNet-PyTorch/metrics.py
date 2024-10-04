import torch


# Funções para cálculo das métricas
def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1e-7):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def accuracy(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    correct = torch.sum(y_true_f == y_pred_f)
    return correct / y_true_f.shape[0]

def precision(y_true, y_pred):
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    return tp / (tp + fp + 1e-7)

def recall(y_true, y_pred):
    tp = torch.sum(y_true * y_pred)
    fn = torch.sum(y_true) - tp
    return tp / (tp + fn + 1e-7)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (2 * prec * rec) / (prec + rec + 1e-7)

def dice_loss(pred, target):
    smooth = 1.0
    num = 2.0 * torch.sum(pred * target) + smooth
    den = torch.sum(pred) + torch.sum(target) + smooth
    loss = 1.0 - (num / den)
    return loss

def iou_loss_smooth(pred, target, smooth=1e-6):
    intersection = torch.sum(pred * target)
    total = torch.sum(pred) + torch.sum(target)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou
