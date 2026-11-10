import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional
from scipy.sparse.linalg import svds



class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 2000,
        loss_function: LossFunction = LossFunction.MSE,
        delta: float = 100,
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []
        self.mse_history = []
        self.delta = delta

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: реализовать функцию предсказания в линейной регрессии
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            gradients = (2 / X.shape[0]) * X.T @ (X @ self.w - y) + 2 * self.l2_coef * self.w
            return gradients
        elif self.loss_function is LossFunction.Huber:
            errors = X @ self.w - y
            derivative = np.clip(errors, -self.delta, self.delta)
            gradient = X.T @ derivative / X.shape[0]
            gradient += 2 * self.l2_coef * self.w
            return gradient
        elif self.loss_function is LossFunction.LogCosh:
            errors = X @ self.w - y
            gradient = X.T @ np.tanh(errors) / X.shape[0]
            gradient += 2 * self.l2_coef * self.w
            return gradient


    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            loss = (
                    np.mean((X @ self.w - y) ** 2)
                    + self.l2_coef * (self.w @ self.w)
            )
            return loss
        elif self.loss_function is LossFunction.Huber:
            errors = X @ self.w - y
            abs_errors = np.abs(errors)

            losses = np.where(
                abs_errors < self.delta,
                0.5 * errors ** 2,
                self.delta * abs_errors - 0.5 * self.delta ** 2
            )

            return (
                    np.mean(losses)
                    + self.l2_coef * (self.w @ self.w)
            )
        elif self.loss_function is LossFunction.LogCosh:
            errors = X @ self.w - y
            losses = np.logaddexp(errors, -errors) - np.log(2.0)

            return (
                    np.mean(losses)
                    + self.l2_coef * (self.w @ self.w)
            )
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y
        self.loss_history = []
        self.mse_history = []

        self.w = np.zeros(X.shape[1], dtype=float)

        if isinstance(self.optimizer, BaseDescent):
            for _ in range(self.max_iter):
                w_prev = self.w.copy()
                self.loss_history.append(self.compute_loss(X, y))
                self.mse_history.append(float(np.mean((X @ self.w - y) ** 2)))
                self.optimizer.step()
                diff = self.w - w_prev
                if diff @ diff < self.tolerance or np.isnan(diff).any():
                    break
            self.loss_history.append(self.compute_loss(X, y))
            self.mse_history.append(float(np.mean((X @ self.w - y) ** 2)))

        elif self.optimizer is None:
            self.w = np.linalg.pinv(X) @ y
        elif self.optimizer == 'SVD':
            U, s, Vt = svds(X, k=4)
            self.w = Vt.T @ np.diag(1 / s) @ U.T @ y
        return self
