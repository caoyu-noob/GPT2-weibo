import torch

class NoamOpt:
    def __init__(self, embeddings_size, warmup, optimizer, linear_schedule=False, lr=None, total_steps=None,
                 apex_level=None, loss_weight=None, extra_module_lr_rate=1.0):
        self.embeddings_size = embeddings_size
        self.warmup = warmup
        self.optimizer = optimizer
        self.linear_schedule = linear_schedule
        self.apex_level = apex_level
        self.lr = lr
        self.total_steps = total_steps
        self.loss_weight = loss_weight
        self.extra_module_lr_rate = extra_module_lr_rate

        self._step = 0

    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except ValueError as e:
            logger.info("Optimizer cannot be loaded from checkpoint: {}".format(e))
        except KeyError as e:
            logger.info("Optimizer cannot be loaded from checkpoint: {}".format(e))

    def backward(self, losses):
        if not isinstance(losses, (tuple, list)):
            losses = [losses]
        if self.loss_weight is None:
            full_loss = sum(losses, 0)
        else:
            full_loss = torch.sum(torch.stack(losses, 0) * torch.exp(self.loss_weight[1])) + torch.sum(
                self.loss_weight[1])

        if self.apex_level is not None:
            try:
                from apex.amp import scale_loss
            except ImportError:
                raise ImportError("Please install apex.")

            for loss_id, loss in enumerate(losses):
                with scale_loss(loss, self.optimizer, loss_id=loss_id) as scaled_loss:
                    scaled_loss.backward()
        else:
            full_loss.backward()
        return full_loss

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate_linear() if self.linear_schedule else self.rate()
        for p in self.optimizer.param_groups:
            if p.__contains__('extra'):
                p['lr'] = rate * self.extra_module_lr_rate
            else:
                p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step

        return self.lr * (self.embeddings_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def rate_linear(self, step=None):
        if step is None:
            step = self._step
        assert self.lr is not None and self.total_steps is not None

        return self.lr * self.warmup_linear(step / self.total_steps, self.warmup)