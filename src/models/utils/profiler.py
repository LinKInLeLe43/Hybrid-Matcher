import contextlib
from pytorch_lightning import profilers
from pytorch_lightning import utilities
import torch


class InferenceProfiler(profilers.SimpleProfiler):
    def __init__(self):
        super().__init__()

        self.start = utilities.rank_zero_only(self.start)
        self.stop = utilities.rank_zero_only(self.stop)
        self.summary = utilities.rank_zero_only(self.summary)

    @contextlib.contextmanager
    def profile(self, action_name: str) -> None:
        try:
            torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            torch.cuda.synchronize()
            self.stop(action_name)
