import torch as th


class ScheduleSampler:
    """ Samples continuous times (in [0, T]) in the forward OU diffusion. """
    def __init__(self, final_time, schedule):
        """
        :param final_time: the final diffusion time T
        :param schedule: can be "uniform" (t sampled uniformly) or "quadratic" (t is the square of a uniform variable)
        """
        self.final_time = final_time
        self.schedule = schedule

    def uniform_to_time(self, u):
        """ Maps the uniform distribution in [0, 1] to the distribution of times in [0, T]. """
        if self.schedule == "uniform":
            pass
        elif self.schedule == "quadratic":
            u = u ** 2
        else:
            raise NotImplementedError(f"unknown schedule sampler: {self.schedule}")
        return self.final_time * u

    def sample(self, batch_size, device):
        """ Sample continuous times for a batch.
        :param batch_size: the number of times to sample.
        :param device: the torch device to save to.
        :return: a tuple (times, weights):
                 - times: a tensor of times.
                 - weights: a tensor of weights to scale the resulting losses (for importance sampling, always 1).
        """
        ts = self.uniform_to_time(th.rand(size=(batch_size,), device=device))
        weights = th.ones(size=(batch_size,), device=device)
        return ts, weights

    def discretize_t(self, num_steps, device):
        """ Returns a (N,) tensor of increasing times in ]0, T] (last time is always T). """
        return self.uniform_to_time((1 + th.arange(num_steps, device=device)) / num_steps)
