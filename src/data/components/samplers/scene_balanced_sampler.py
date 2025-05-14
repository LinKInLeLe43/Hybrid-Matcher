import torch
from torch.utils import data


class SceneBalancedSampler(data.Sampler):
    def __init__(
        self,
        concat_dataset: data.ConcatDataset,
        num_samples_per_scene: int,
        replacement: bool = True,
        shuffle: bool = True,
        repeat: int = 1,
        seed: int = 66
    ) -> None:
        super().__init__(self)
        self.concat_dataset = concat_dataset
        self.num_samples_per_scene = num_samples_per_scene
        self.replacement = replacement
        self.shuffle = shuffle
        self.repeat = repeat

        self.subsets_count = len(concat_dataset.datasets)
        self.samples_count = (repeat * self.subsets_count *
                              num_samples_per_scene)
        self.generator = torch.manual_seed(seed)

    def __iter__(self):
        idxes = []
        low = 0
        for i in range(self.subsets_count):
            high = self.concat_dataset.cumulative_sizes[i]
            if self.replacement:
                idxes_per_subset = torch.randint(
                    low, high, (self.num_samples_per_scene,),
                    generator=self.generator)
            else:
                subset_samplers_count = high - low
                idxes_per_subset = torch.randperm(
                    subset_samplers_count, generator=self.generator)
                idxes_per_subset += low
                if subset_samplers_count >= self.num_samples_per_scene:
                    idxes_per_subset = (
                        idxes_per_subset[:self.num_samples_per_scene])
                else:
                    padding_count = (
                        self.num_samples_per_scene - subset_samplers_count)
                    padding_idxes_per_subset = torch.randint(
                        low, high, (padding_count,), generator=self.generator)
                    idxes_per_subset = torch.cat([idxes_per_subset,
                                                  padding_idxes_per_subset])
            idxes.append(idxes_per_subset)
            low = high
        idxes = torch.cat(idxes)

        count = len(idxes)
        if self.shuffle:
            subidxes = torch.randperm(count, generator=self.generator)
            idxes = idxes[subidxes]

        if self.repeat > 1:
            if self.shuffle:
                repeat_idxes = []
                for _ in range(self.repeat - 1):
                    subidxes = torch.randperm(count, generator=self.generator)
                    repeat_idxes.append(idxes.clone()[subidxes])
            else:
                repeat_idxes = [idxes.clone() for _ in range(self.repeat - 1)]
            idxes = torch.cat([idxes, *repeat_idxes])
        it = iter(idxes.tolist())
        return it

    def __len__(self) -> int:
        return self.samples_count
