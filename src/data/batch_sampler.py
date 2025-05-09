from typing import Generator, List, Optional

import numpy as np
import torch

from .dataset import GameHdf5Dataset, Dataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            dataset: Dataset,
            rank: int,
            world_size: int,
            batch_size: int,
            seq_length: int,
            sample_weights: Optional[List[float]] = None,
            can_sample_beyond_end: bool = False,
            autoregressive_obs: int = None,
            initial_num_consecutive_page_count: int = 1
    ) -> None:
        super().__init__(dataset)
        assert isinstance(dataset, (Dataset, GameHdf5Dataset))
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.can_sample_beyond_end = can_sample_beyond_end
        self.autoregressive_obs = autoregressive_obs
        self.num_consecutive_batches = initial_num_consecutive_page_count

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        segments = None
        current_iter = 0

        while True:
            if current_iter == 0:
                segments = self.sample()
            else:
                segments = self.next(segments)

            current_iter = (current_iter + 1) % self.num_consecutive_batches
            yield segments

    def next(self, segments: List[SegmentId]):
        return [
            SegmentId(segment.episode_id, segment.stop, segment.stop + self.autoregressive_obs, False)
            for segment in segments
        ]

    def sample(self) -> List[SegmentId]:
        total_length = self.seq_length + (self.num_consecutive_batches - 1) * self.autoregressive_obs

        num_episodes = self.dataset.num_episodes

        if (self.sample_weights is None) or num_episodes < len(self.sample_weights):
            weights = self.dataset.lengths / self.dataset.num_steps
        else:
            weights = self.sample_weights
            num_weights = len(self.sample_weights)
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        episode_lengths = self.dataset.lengths[episodes_partition]
        valid_mask = episode_lengths > total_length # valid episodes must be long enough for autoregressvie generation
        episodes_partition = episodes_partition[valid_mask]

        weights = np.array(weights[self.rank::self.world_size])
        weights = weights[valid_mask]

        max_eps = self.batch_size
        episode_ids = np.random.choice(episodes_partition, size=max_eps, replace=True, p=weights / weights.sum())
        episode_ids = episode_ids.repeat(self.batch_size // max_eps)

        # choose a random timestamp at the dataset
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])
        # compute total context size + autoregressive generation length

        # the stops of the first page can be at most the length of the training example minus the autoregressive generation frames in the next pages
        stops = np.minimum(
            self.dataset.lengths[episode_ids] - (self.num_consecutive_batches - 1) * self.seq_length,
            timesteps + 1 + np.random.randint(0, total_length, len(timesteps))
        )
        # stops must be longer than the initial context + first page prediction size
        stops = np.maximum(stops, self.seq_length)
        # starts is stops minus the initial context and the first page prediction size
        starts = stops - self.seq_length

        return [SegmentId(*x, True) for x in zip(episode_ids, starts, stops)]
