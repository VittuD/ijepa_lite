from __future__ import annotations


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)
