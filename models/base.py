from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaseModel(ABC):
    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def train_on_fold(self, training_loader, validation_loader):
        """
        Train the model on the given training data and evaluate on validation data.
        Should return accuracy, precision, recall, F1 score, and the trained model instance.

        Returns:
            acc (float): Accuracy of the model on validation data.
            prec (float): Precision of the model on validation data.
            rec (float): Recall of the model on validation data.
            f1 (float): F1 score of the model on validation data.
            trained_model (object): The trained model instance.
        """
        pass

def worm_level_aggregation(worm_ids, probs, true_labels, threshold=0.5):
    results_df = pd.DataFrame(
        {"Worm_ID": worm_ids, "Prob_Segment": probs, "True_Label": true_labels}
    )
    worm_results = results_df.groupby("Worm_ID").agg(
        {"Prob_Segment": "mean", "True_Label": "first"}
    )
    worm_preds = (worm_results["Prob_Segment"] > threshold).astype(int)
    worm_truth = worm_results["True_Label"]
    return worm_preds, worm_truth


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec, f1
