class AverageMeter:
    def __init__(
        self,
    ):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / (self.count + 1e-3)