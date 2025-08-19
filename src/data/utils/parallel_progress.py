import contextlib
from joblib import parallel
from rich import progress
import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm.tqdm) -> tqdm.tqdm:
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, out):
            tqdm_object.update(n=self.batch_size)
            super().__call__(out)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


@contextlib.contextmanager
def rich_joblib(rich_object: progress.Progress) -> progress.Progress:
    class RichBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, out):
            if len(rich_object.task_ids) > 1:
                raise ValueError("")

            rich_object.update(rich_object.task_ids[0], advance=self.batch_size)
            super().__call__(out)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = RichBatchCompletionCallback
    try:
        yield rich_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        rich_object.stop()
