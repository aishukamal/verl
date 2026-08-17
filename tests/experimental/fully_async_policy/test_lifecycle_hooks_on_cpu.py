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

"""Unit tests for the fully async trainer lifecycle hooks and registry (no GPU required)."""

import asyncio
from types import SimpleNamespace

import pytest

from verl.experimental.fully_async_policy.fully_async_trainer import (
    TRAINER_REGISTRY,
    FullyAsyncTrainer,
    get_trainer_cls,
    register_trainer,
)


def _bare_trainer(cls=FullyAsyncTrainer, **attrs):
    """Instantiate ``cls`` without running __init__ and set the given attributes."""
    trainer = object.__new__(cls)
    for key, value in attrs.items():
        setattr(trainer, key, value)
    return trainer


class _Recorder(FullyAsyncTrainer):
    """Subclass recording hook invocations; init/model plumbing stubbed out."""

    def _init_resource_pools(self):
        self.events.append("init_resource_pools")

    def _create_worker_classes(self):
        self.events.append("create_worker_classes")

    def _init_worker_groups(self):
        self.events.append("init_worker_groups")

    def _init_models(self):
        self.events.append("init_models")

    def on_init_workers_begin(self):
        self.events.append("on_init_workers_begin")

    async def on_init_workers_end(self):
        self.events.append("on_init_workers_end")

    def on_update_weights_begin(self):
        self.events.append("on_update_weights_begin")

    def on_update_weights_end(self, synced):
        self.events.append(("on_update_weights_end", synced))

    def on_sample_end(self):
        self.events.append("on_sample_end")

    def on_exception(self, point, exc):
        self.events.append(("on_exception", point, exc))


def test_base_hooks_are_noops():
    trainer = _bare_trainer()
    assert trainer.on_init_workers_begin() is None
    assert trainer.on_init_workers_end() is None
    assert trainer.on_sample_end() is None
    assert trainer.on_update_weights_begin() is None
    assert trainer.on_update_weights_end(synced=True) is None
    assert trainer.on_save_checkpoint(force=False) is True
    assert trainer.on_exception("fit", RuntimeError()) is None


def test_init_workers_fires_begin_and_end_hooks_in_order():
    trainer = _bare_trainer(_Recorder, events=[])
    asyncio.run(trainer.init_workers())
    # on_init_workers_end is a coroutine function: awaitable results are awaited.
    assert trainer.events == [
        "on_init_workers_begin",
        "init_resource_pools",
        "create_worker_classes",
        "init_worker_groups",
        "init_models",
        "on_init_workers_end",
    ]


def test_init_workers_failure_fires_on_exception_and_reraises():
    class Failing(_Recorder):
        def _init_worker_groups(self):
            raise RuntimeError("pool exhausted")

    trainer = _bare_trainer(Failing, events=[])
    with pytest.raises(RuntimeError, match="pool exhausted"):
        asyncio.run(trainer.init_workers())
    assert trainer.events[0] == "on_init_workers_begin"
    exception_events = [e for e in trainer.events if isinstance(e, tuple) and e[0] == "on_exception"]
    assert len(exception_events) == 1
    assert exception_events[0][1] == "init_workers"
    assert "on_init_workers_end" not in trainer.events


def test_hook_exceptions_propagate():
    class Aborting(FullyAsyncTrainer):
        def on_init_workers_begin(self):
            raise RuntimeError("phase not granted")

    trainer = _bare_trainer(Aborting, _last_on_exception_exc=None)
    with pytest.raises(RuntimeError, match="phase not granted"):
        asyncio.run(trainer.init_workers())


def test_dispatch_on_exception_fires_once_and_suppresses_hook_errors():
    class BrokenHook(FullyAsyncTrainer):
        def __init__(self):  # noqa: super-init-not-called - bare instance for hook tests
            self.calls = 0

        def on_exception(self, point, exc):
            self.calls += 1
            raise ValueError("hook bug")

    trainer = BrokenHook()
    original = RuntimeError("training failed")
    # Hook errors are suppressed and the same exception is dispatched only once.
    asyncio.run(trainer._dispatch_on_exception("init_workers", original))
    asyncio.run(trainer._dispatch_on_exception("fit", original))
    assert trainer.calls == 1
    asyncio.run(trainer._dispatch_on_exception("fit", RuntimeError("another")))
    assert trainer.calls == 2


def test_fit_update_weights_noop_path_fires_end_with_synced_false():
    trainer = _bare_trainer(_Recorder, events=[], local_trigger_step=2)
    result = asyncio.run(trainer._fit_update_weights())
    assert result is None
    assert trainer.events == ["on_update_weights_begin", ("on_update_weights_end", False)]


def test_fit_generate_fires_on_sample_end_after_batch_assembly():
    batch = SimpleNamespace(meta_info={})

    class Sampler(_Recorder):
        async def _get_samples_from_queue(self):
            self.events.append("get_samples_from_queue")
            return 0, batch

    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.7)))
    trainer = _bare_trainer(Sampler, events=[], metrics={}, timing_raw={}, config=config)
    result = asyncio.run(trainer._fit_generate())
    assert result is batch
    assert trainer.events == ["get_samples_from_queue", "on_sample_end"]
    assert batch.meta_info["temperature"] == 0.7


def test_save_checkpoint_veto_skips_save():
    class Vetoing(FullyAsyncTrainer):
        def on_save_checkpoint(self, force):
            return False

    # No checkpoint plumbing attributes are set: reaching past the veto would raise.
    trainer = _bare_trainer(Vetoing, current_param_version=3)
    assert trainer._fit_save_checkpoint(force=True) is None


def test_save_checkpoint_default_proceeds():
    # Default hook returns True; the version guard then short-circuits the save.
    trainer = _bare_trainer(current_param_version=3, last_ckpt_version=3)
    assert trainer._fit_save_checkpoint() is None


def test_registry_contains_builtin_trainer():
    assert get_trainer_cls("fully_async") is FullyAsyncTrainer


def test_register_and_get_trainer_cls():
    @register_trainer("_test_recorder")
    class Registered(_Recorder):
        pass

    try:
        assert get_trainer_cls("_test_recorder") is Registered
        # Re-registering the same class under the same name is a no-op.
        assert register_trainer("_test_recorder")(Registered) is Registered
    finally:
        TRAINER_REGISTRY.pop("_test_recorder", None)


def test_register_trainer_rejects_duplicate_name():
    @register_trainer("_test_dup")
    class First(FullyAsyncTrainer):
        pass

    try:
        with pytest.raises(ValueError, match="already registered"):

            @register_trainer("_test_dup")
            class Second(FullyAsyncTrainer):
                pass

    finally:
        TRAINER_REGISTRY.pop("_test_dup", None)


def test_register_trainer_rejects_non_subclass():
    with pytest.raises(TypeError, match="FullyAsyncTrainer subclass"):
        register_trainer("_test_bad")(object)
    assert "_test_bad" not in TRAINER_REGISTRY


def test_get_trainer_cls_unknown_name_lists_available():
    with pytest.raises(ValueError, match="Unknown trainer '_no_such'.*fully_async"):
        get_trainer_cls("_no_such")
