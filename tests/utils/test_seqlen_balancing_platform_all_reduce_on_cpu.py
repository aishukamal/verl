# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DP-wide reductions of ``rearrange_micro_batches`` go through ``PlatformBase.all_reduce``.

No process group is created: ``dist.is_initialized`` is stubbed and a fake platform records the calls.
"""

from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from verl import DataProto
from verl.plugin.platform.platform_base import PlatformBase
from verl.utils import seqlen_balancing
from verl.utils.seqlen_balancing import rearrange_micro_batches


class _RecordingPlatform:
    """Single-rank semantics: records (op, group) and leaves the tensor as it is."""

    def __init__(self):
        self.calls = []

    def all_reduce(self, tensor, op, group=None):
        self.calls.append((op, group, tensor.clone()))


class _RaisingCountPlatform(_RecordingPlatform):
    """Pretends a peer rank asked for ``count`` micro-batches on the first MAX reduction."""

    def __init__(self, count):
        super().__init__()
        self.count = count

    def all_reduce(self, tensor, op, group=None):
        super().all_reduce(tensor, op, group)
        if len(self.calls) == 1:
            tensor.fill_(self.count)


def _batch():
    input_ids = torch.zeros((8, 7), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return DataProto.from_single_dict({"input_ids": input_ids, "attention_mask": attention_mask}).batch


def _run(platform, **kwargs):
    group = object()
    with (
        mock.patch.object(seqlen_balancing, "get_platform", return_value=platform),
        mock.patch.object(seqlen_balancing.dist, "is_initialized", return_value=True),
        mock.patch.object(seqlen_balancing, "get_device_name", return_value="cpu"),
    ):
        micro_batches, _ = rearrange_micro_batches(
            _batch(), max_token_len=14, dp_group=group, same_micro_num_in_dp=True, **kwargs
        )
    return micro_batches, group


def test_platform_base_default_delegates_to_torch_distributed():
    tensor, group = torch.tensor([3]), object()
    with mock.patch.object(dist, "all_reduce") as all_reduce:
        PlatformBase.all_reduce(SimpleNamespace(), tensor, op=dist.ReduceOp.MAX, group=group)
    all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.MAX, group=group)


def test_reductions_are_routed_through_the_platform_hook():
    platform = _RecordingPlatform()
    micro_batches, group = _run(platform)

    # micro-batch count (MAX), fatal constraints (MAX), within-limit flag (MIN), in this order.
    assert [op for op, _, _ in platform.calls] == [dist.ReduceOp.MAX, dist.ReduceOp.MAX, dist.ReduceOp.MIN]
    assert all(call_group is group for _, call_group, _ in platform.calls)
    assert platform.calls[0][2].tolist() == [4]  # 8 x 7 tokens, 14 per micro-batch
    assert platform.calls[1][2].tolist() == [7, -8]  # max group token len, -num_groups
    assert platform.calls[2][2].tolist() == [1]
    assert len(micro_batches) == 4
    assert all(mb["attention_mask"].sum().item() <= 14 for mb in micro_batches)


def test_reduced_value_is_used():
    platform = _RaisingCountPlatform(count=8)
    micro_batches, _ = _run(platform)
    assert len(micro_batches) == 8


def test_no_hook_calls_without_a_dp_group():
    platform = _RecordingPlatform()
    with mock.patch.object(seqlen_balancing, "get_platform", return_value=platform):
        rearrange_micro_batches(_batch(), max_token_len=14)
    assert platform.calls == []
