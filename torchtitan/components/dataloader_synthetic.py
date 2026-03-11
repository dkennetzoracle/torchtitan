# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic (random-token) dataloader for benchmarking and health checks.

Generates random integer token batches without any real dataset download,
tokenizer, or preprocessing. Suitable for TFLOPs / MFU benchmarking.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader


class SyntheticDataLoader(BaseDataLoader):
    """Yields infinite random token batches with no external dependencies.

    Generates ``({"input": tokens}, labels)`` tuples where both tensors have
    shape ``[local_batch_size, seq_len]`` and dtype ``torch.long``.
    The input/label split mirrors the causal-LM convention used by
    ``HuggingFaceTextDataset`` (input = tokens[:-1], label = tokens[1:]).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        vocab_size: int = 128256
        """Vocabulary size for random token generation (default: Llama 3 vocab size)."""

    def __init__(
        self,
        config: Config,
        *,
        dp_rank: int,
        dp_world_size: int,
        tokenizer: Any,  # accepted for interface compatibility, not used
        seq_len: int,
        local_batch_size: int,
        **kwargs: Any,
    ) -> None:
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.vocab_size = config.vocab_size
        self._step = 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            # Draw [batch, seq_len+1] tokens then split into causal input/label pair.
            tokens = torch.randint(
                0,
                self.vocab_size,
                (self.local_batch_size, self.seq_len + 1),
                dtype=torch.long,
            )
            yield {"input": tokens[:, :-1]}, tokens[:, 1:]
            self._step += 1

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict.get("step", 0)
