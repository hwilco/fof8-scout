import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


class BetaCalibrator:
    """
    Beta Calibrator for transforming classifier probabilities.

    Uses a logistic regression on log-transformed probabilities:
    logit(p_cal) = a * log(p) + b * (-log(1-p)) + c
    """

    def __init__(self, eps: float = 1e-10) -> None:
        """
        Initialize the BetaCalibrator.

        Args:
            eps: Small epsilon to avoid log(0) or log(1) errors.
        """
        self.eps = eps
        self.model = LogisticRegression(solver="lbfgs", C=1e10)

    def _transform_features(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities into Beta-space features.

        Args:
            y_prob: Raw probabilities.

        Returns:
            Transformed feature matrix.
        """
        p = np.clip(y_prob, self.eps, 1 - self.eps)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BetaCalibrator":
        """
        Fit the calibrator.

        Args:
            y_prob: OOF probabilities from the classifier.
            y_true: True binary labels.

        Returns:
            Self for chaining.
        """
        X_beta = self._transform_features(y_prob)
        self.model.fit(X_beta, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transform probabilities using the fitted calibrator.

        Args:
            y_prob: Raw probabilities to transform.

        Returns:
            Calibrated probabilities.
        """
        X_beta = self._transform_features(y_prob)
        return self.model.predict_proba(X_beta)[:, 1]


def run_calibration_audit(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Runs a formal calibration audit using Cox Slope and Intercept,
    Brier score, and Spiegelhalter's Z-test.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary containing audit metrics.
    """
    # 1. Cox Calibration Audit
    eps = 1e-10
    p_clip = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)

    lr = LogisticRegression(C=1e10)
    lr.fit(logit_p, y_true)

    alpha = float(np.ravel(lr.intercept_)[0])
    beta = float(np.ravel(lr.coef_)[0])

    # 2. Spiegelhalter's Z-test
    brier = np.mean((y_true - y_prob) ** 2)
    expected_brier = np.mean(y_prob * (1 - y_prob))
    variance_brier = np.sum(((1 - 2 * y_prob) ** 2) * y_prob * (1 - y_prob)) / (len(y_prob) ** 2)

    z_score = (brier - expected_brier) / np.sqrt(variance_brier)
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    return {
        "cox_intercept": alpha,
        "cox_slope": beta,
        "spiegelhalter_z": float(z_score),
        "spiegelhalter_p": float(p_value),
        "brier_score": float(brier),
    }
