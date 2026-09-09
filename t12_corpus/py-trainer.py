def helper_1(x: int) -> int:
    return x * 1

def helper_2(x: int) -> int:
    return x * 2

def helper_3(x: int) -> int:
    return x * 3

def helper_4(x: int) -> int:
    return x * 4

def helper_5(x: int) -> int:
    return x * 5

class Trainer:
    def __init__(self):
        self.loss = 0.0

    def fit(self, epochs: int) -> None:
        for _ in range(epochs):
            self.loss += 0.1

