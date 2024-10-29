import torch

<<<<<<< HEAD

class NoamOpt:
    """
    Copied from https://nlp.seas.harvard.edu/2018/04/03/attention.html#hardware-and-schedule

    A wrapper class for the Adam optimizer (or others) that implements learning rate scheduling.

    """
=======
class NoamOpt:
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
<<<<<<< HEAD
        """
        Update parameters and rate"
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


=======
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
def get_std_opt(model):
    return NoamOpt(
        model.encoder.hidden_dim,
        2,
        4000,
<<<<<<< HEAD
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
=======
        torch.optim.Adam(
            model.parameters(),
            lr = 0,
            betas = (0.9, 0.98),
            eps = 1e-9
        ),
>>>>>>> dab5c9249314359e4f2c40ef3899de33729105da
    )