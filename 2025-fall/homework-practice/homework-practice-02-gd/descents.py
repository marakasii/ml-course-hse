import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p

# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        diff = lr * gradient

        self.model.w -= diff
        return -diff

class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        sample = np.random.randint(self.model.X_train.shape[0], size=self.batch_size)
        X_train = self.model.X_train[sample]
        y_train = self.model.y_train[sample]
        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)
        diff = lr * gradient

        self.model.w -= diff
        return -diff


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros(shape=(num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
        # TODO: реализовать SAG
        pos = self.iteration % num_objects
        new = self.model.compute_gradients(X_train[pos:pos + 1], y_train[pos:pos + 1])

        self.grad_sum += new - self.grad_memory[pos]
        self.grad_memory[pos] = new

        lr = self.lr_schedule.get_lr(self.iteration)
        diff = lr * (self.grad_sum / num_objects)
        self.model.w -= diff
        return -diff



class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        # TODO: реализовать градиентный спуск с моментумом
        X_train = self.model.X_train
        y_train = self.model.y_train

        if self.velocity is None:
            self.velocity = 0

        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)
        diff = lr * gradient

        self.velocity = self.velocity * self.beta + diff
        self.model.w -= self.velocity
        return -self.velocity


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        # TODO: реализовать Adam по формуле из ноутбука
        X_train = self.model.X_train
        y_train = self.model.y_train

        if self.m is None:
            self.m = self.v = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients(X_train, y_train)

        lr = self.lr_schedule.get_lr(self.iteration)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_avg = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_avg = self.v / (1 - self.beta2 ** (self.iteration + 1))

        diff = lr / (np.sqrt(v_avg) + self.eps) * m_avg

        self.model.w -= diff
        return -diff