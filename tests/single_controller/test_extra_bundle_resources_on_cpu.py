# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for per-pool extra placement group bundle resources (no GPU/cluster required)."""

import pytest
from omegaconf import OmegaConf

from verl.single_controller.ray.base import (
    RayResourcePool,
    apply_extra_bundle_resources_config,
    resolve_extra_bundle_resources,
    set_extra_bundle_resources_by_pool,
)


@pytest.fixture(autouse=True)
def _clear_mapping():
    yield
    set_extra_bundle_resources_by_pool(None)


def test_resolve_longest_prefix_match():
    set_extra_bundle_resources_by_pool(
        {
            "trainer_pool": {"trainer_node": 1},
            "rollout_pool": {"rollout_node": 1},
            "rollout_pool_reward": {"reward_node": 1},
        }
    )
    assert resolve_extra_bundle_resources("trainer_pool") == {"trainer_node": 1.0}
    assert resolve_extra_bundle_resources("rollout_pool_0") == {"rollout_node": 1.0}
    assert resolve_extra_bundle_resources("rollout_pool_reward_0") == {"reward_node": 1.0}
    assert resolve_extra_bundle_resources("some_other_pool") is None


def test_resolve_with_empty_mapping():
    set_extra_bundle_resources_by_pool(None)
    assert resolve_extra_bundle_resources("trainer_pool") is None


def test_setter_coerces_omegaconf_sections():
    cfg = OmegaConf.create({"trainer_pool": {"trainer_node": 1}})
    set_extra_bundle_resources_by_pool(cfg)
    resolved = resolve_extra_bundle_resources("trainer_pool")
    assert resolved == {"trainer_node": 1.0}
    assert isinstance(resolved, dict)
    assert all(isinstance(v, float) for v in resolved.values())


def test_apply_config_section():
    config = OmegaConf.create({"ray_pg_extra_resources": {"rollout_pool": {"rollout_node": 2}}})
    apply_extra_bundle_resources_config(config)
    assert resolve_extra_bundle_resources("rollout_pool_1") == {"rollout_node": 2.0}


def test_apply_config_noop_without_section():
    set_extra_bundle_resources_by_pool({"trainer_pool": {"trainer_node": 1}})
    apply_extra_bundle_resources_config(OmegaConf.create({}))
    apply_extra_bundle_resources_config(OmegaConf.create({"ray_pg_extra_resources": None}))
    apply_extra_bundle_resources_config(None)
    # Absent/None sections must not clear an existing mapping.
    assert resolve_extra_bundle_resources("trainer_pool") == {"trainer_node": 1.0}


def test_pool_explicit_resources_stored():
    pool = RayResourcePool(
        process_on_nodes=[1],
        use_gpu=True,
        name_prefix="trainer_pool",
        extra_bundle_resources={"trainer_node": 1},
    )
    assert pool.extra_bundle_resources == {"trainer_node": 1}

    default_pool = RayResourcePool(process_on_nodes=[1], use_gpu=True, name_prefix="trainer_pool")
    assert default_pool.extra_bundle_resources is None
