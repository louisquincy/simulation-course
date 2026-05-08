import time

class MultiKongGen:
    """Собственный ГПСЧ на основе линейного конгруэнтного метода."""
    def __init__(self, seed=None):
        self.M = 2 ** 63
        self.beta = 2 ** 32 + 3
        if seed is None:
            seed = time.time_ns() % self.M
        self.x0 = seed

    def anabios(self):
        """Возвращает случайное число в [0, 1)."""
        self.x0 = (self.beta * self.x0) % self.M
        return self.x0 / self.M