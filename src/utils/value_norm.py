import torch as th


class ValueNorm:
    def __init__(self, shape, beta=0.99999, epsilon=1e-5, device="cpu"):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.beta = beta
        self.epsilon = epsilon
        self.device = device

        self.running_mean = th.zeros(self.shape, dtype=th.float32, device=device)
        self.running_mean_sq = th.zeros(self.shape, dtype=th.float32, device=device)
        self.debiasing_term = th.zeros(1, dtype=th.float32, device=device)

    def update(self, values, mask=None):
        values = values.detach().to(self.running_mean.device, dtype=th.float32)
        flat_values = values.reshape(-1, *self.shape)

        if mask is not None:
            mask = mask.detach().to(self.running_mean.device)
            flat_mask = mask.reshape(-1)
            valid = flat_mask > 0
            if valid.sum() == 0:
                return
            flat_values = flat_values[valid]
        elif flat_values.numel() == 0:
            return

        batch_mean = flat_values.mean(dim=0)
        batch_mean_sq = (flat_values ** 2).mean(dim=0)

        weight = 1.0 - self.beta
        self.running_mean.mul_(self.beta).add_(batch_mean * weight)
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * weight)
        self.debiasing_term.mul_(self.beta).add_(weight)

    def normalize(self, values):
        mean, var = self.running_mean_var()
        return (values - mean) / th.sqrt(var)

    def denormalize(self, values):
        mean, var = self.running_mean_var()
        return values * th.sqrt(var) + mean

    def running_mean_var(self):
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        var = (mean_sq - mean ** 2).clamp(min=self.epsilon)
        return mean, var

    def state_dict(self):
        return {
            "shape": self.shape,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "running_mean": self.running_mean,
            "running_mean_sq": self.running_mean_sq,
            "debiasing_term": self.debiasing_term,
        }

    def load_state_dict(self, state_dict):
        self.running_mean = state_dict["running_mean"].to(self.running_mean.device)
        self.running_mean_sq = state_dict["running_mean_sq"].to(self.running_mean_sq.device)
        self.debiasing_term = state_dict["debiasing_term"].to(self.debiasing_term.device)

    def to(self, device):
        self.device = device
        self.running_mean = self.running_mean.to(device)
        self.running_mean_sq = self.running_mean_sq.to(device)
        self.debiasing_term = self.debiasing_term.to(device)
        return self
