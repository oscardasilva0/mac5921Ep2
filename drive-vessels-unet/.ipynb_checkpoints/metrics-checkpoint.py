import torch
import numpy as np

class Metrics:

    def __init__(self):
        pass

    def calculate_dice_score(self, output, target):
        smooth = 1.0
        intersection = (output * target).sum()
        return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    def calculate_precision(self, output, target):
        smooth = 1.0
        tp = (output * target).sum()
        fp = (output * (1 - target)).sum()
        return (tp + smooth) / (tp + fp + smooth)

    def calculate_recall(self, output, target):
        smooth = 1.0
        tp = (output * target).sum()
        fn = (target * (1 - output)).sum()
        return (tp + smooth) / (tp + fn + smooth)

    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    def calculate_iou(self, output, target):
        intersection = (output * target).sum()
        union = (output + target).sum() - intersection
        return (intersection + 1e-6) / (union + 1e-6)  # Adicione um valor pequeno para evitar divisão por zero