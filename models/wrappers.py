from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TrainingWrapper(ABC):
    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def train_on_fold(self, training_loader, validation_loader):
        """
        Train the model on the given training data and evaluate on validation data.
        Should return accuracy, precision, recall, F1 score, and the trained model instance.

        Returns:
            measures (dict): A dictionary containing measures of interest, like accuracy, precision, recall, and F1 score.
            trained_model (object): The trained model instance.
        """
        pass

class BenchmarkWrapper(ABC):
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None

    @abstractmethod
    def load(self, path):
        """
        Load a trained model checkpoint from the given path.
        """
        pass

    @abstractmethod
    def benchmark(self, test_loader):
        """
        Evaluate the loaded model on the provided test data loader.
        Returns measures of interest (e.g., accuracy, loss, etc.) in a dictionary.
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
