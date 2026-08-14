# IMPORTS
import numpy as np

# LOCAL IMPORTS
from config.configs import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomLogisticRegression:
    def __init__(self, learning_rate=Config.LEARNING_RATE, epochs=Config.EPOCHS,
                 class_weight=Config.CLASS_WEIGHT, reg=Config.REGULARIZATION,
                 reg_lambda=Config.REG_LAMBDA):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.reg = reg
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # clip to prevent overflow in exp
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))

    def _compute_loss(self, y, y_pred, m):
        # compute sample weights for class imbalance
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            weight_map = {cls: m / (len(unique) * count) for cls, count in zip(unique, counts)}
            sample_weights = np.array([weight_map[label] for label in y])
        elif isinstance(self.class_weight, dict):
            sample_weights = np.array([self.class_weight[label] for label in y])
        else:
            sample_weights = np.ones(m)

        # weighted cross-entropy loss
        eps = 1e-9  # small epsilon to avoid log(0)
        loss = -np.mean(sample_weights * (y * np.log(y_pred + eps) +
                                          (1 - y) * np.log(1 - y_pred + eps)))

        # add regularization
        if self.reg == 'l2':
            loss += (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)
        elif self.reg == 'l1':
            loss += (self.reg_lambda / m) * np.sum(np.abs(self.weights))

        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.loss_history = []

        logger.info("starting training: %d samples, %d features, %d epochs",
                   m, n, self.epochs)

        for epoch in range(self.epochs):
            # forward pass: compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # compute sample weights for gradient calculation
            if self.class_weight == 'balanced':
                unique, counts = np.unique(y, return_counts=True)
                weight_map = {cls: m / (len(unique) * count) for cls, count in zip(unique, counts)}
                sample_weights = np.array([weight_map[label] for label in y])
            elif isinstance(self.class_weight, dict):
                sample_weights = np.array([self.class_weight[label] for label in y])
            else:
                sample_weights = np.ones(m)

            # compute gradients with sample weights
            dw = (1 / m) * np.dot(X.T, sample_weights * (y_pred - y))
            db = (1 / m) * np.sum(sample_weights * (y_pred - y))

            # add regularization gradients
            if self.reg == 'l2':
                dw += (self.reg_lambda / m) * self.weights
            elif self.reg == 'l1':
                dw += (self.reg_lambda / m) * np.sign(self.weights)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # compute and store loss
            loss = self._compute_loss(y, y_pred, m)
            self.loss_history.append(loss)

            # log progress every 100 epochs
            if epoch % 100 == 0:
                logger.info("epoch %d/%d - loss: %.6f", epoch, self.epochs, loss)

        logger.info("training complete! final loss: %.6f", self.loss_history[-1])
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("model has not been fitted yet. call fit() first.")

        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)