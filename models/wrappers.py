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
        Returns trajectories of each sample.

        Returns:
            dict:
                predictions (list): List of predicted probabilities for each sample.
                variances (list): List of variances associated with each prediction.
                interpretability_score: A measure of how interpretable the model's predictions are.
        """
        pass

class VisualizationWrapper(ABC):
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
    def get_trajectory_predictions(self, data_tensor, total_segments):
        """
        Run inference on a single trajectory step-by-step.
        
        Args:
            data_tensor: Tensor of shape (T, ...), representing the full sample
            total_segments: The number of valid segments (T_actual).
            
        Returns:
            predictions (list or array): List of predicted remaining lifetime for each step t=1..T_actual.
            variances (list or array): List of variances for each prediction.
            custom_data (dict): A dictionary with model-specific data (e.g. 's_weights', 'v_weights' for attention).
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
