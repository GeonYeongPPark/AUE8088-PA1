from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # States for true positives, false positives, and false negatives for each class
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape

        for cls in range(self.num_classes):
            tp = torch.sum((preds == cls) & (target == cls))
            fp = torch.sum((preds == cls) & (target != cls))
            fn = torch.sum((preds != cls) & (target == cls))

            self.true_positives[cls] += tp
            self.false_positives[cls] += fp
            self.false_negatives[cls] += fn

    def compute(self):
        eps = 1e-6
        precision = self.true_positives / (self.true_positives + self.false_positives + eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        
        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)
        
        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct
        
        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
