from LAMDA_SSL.Base.LambdaLR import LambdaLR

class DAScheduler(LambdaLR):
    def __init__(self, lr_decay=3e-4, lr_gamma=0.75):
        self.lr_decay=lr_decay
        self.lr_gamma=lr_gamma
        super().__init__(lr_lambda=self._lr_lambda)

    def _lr_lambda(self, current_step):
        return (1. + self.lr_gamma * float(current_step)) ** (-self.lr_decay)