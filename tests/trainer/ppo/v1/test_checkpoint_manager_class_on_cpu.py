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
"""``PPOTrainer._build_checkpoint_manager`` honours ``rollout.checkpoint_manager_class``.

Runs without Ray, workers or an accelerator: the trainer object is built with ``__new__`` and only
the attributes the manager construction reads are set.
"""

from types import SimpleNamespace

from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class FakeCheckpointEngineManager:
    """Stands in for a plugin-provided ``CheckpointEngineManager`` subclass, loaded by FQN."""

    instances: list = []

    def __init__(self, config, actor_wg, replicas):
        self.config = config
        self.actor_wg = actor_wg
        self.replicas = replicas
        FakeCheckpointEngineManager.instances.append(self)


def _trainer(checkpoint_manager_class=None, backend="raiden"):
    trainer = _StubTrainer.__new__(_StubTrainer)
    rollout = {"checkpoint_engine": {"_target_": "verl.workers.config.CheckpointEngineConfig", "backend": backend}}
    if checkpoint_manager_class is not None:
        rollout["checkpoint_manager_class"] = checkpoint_manager_class
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": rollout}})
    trainer.actor_rollout_wg = SimpleNamespace(name="actor_wg")
    trainer.llm_server_manager = SimpleNamespace(get_replicas=lambda: ["replica0"])
    return trainer


def test_custom_manager_class_is_loaded_and_keeps_the_configured_backend():
    FakeCheckpointEngineManager.instances.clear()
    trainer = _trainer(checkpoint_manager_class=f"{__name__}.FakeCheckpointEngineManager")

    manager = trainer._build_checkpoint_manager()

    assert isinstance(manager, FakeCheckpointEngineManager)
    assert FakeCheckpointEngineManager.instances == [manager]
    assert manager.config.backend == "raiden"  # not overwritten to "naive"
    assert manager.actor_wg is trainer.actor_rollout_wg
    assert manager.replicas == ["replica0"]


def test_default_manager_still_forces_the_naive_backend(monkeypatch):
    built = []

    class RecordingManager:
        def __init__(self, config, actor_wg, replicas):
            built.append(config.backend)

    # The default path constructs the module's CheckpointEngineManager name, so substituting it
    # observes exactly what the stock manager would be given.
    monkeypatch.setattr(trainer_base, "CheckpointEngineManager", RecordingManager)
    manager = _trainer(backend="raiden")._build_checkpoint_manager()

    assert isinstance(manager, RecordingManager)
    assert built == ["naive"]


def test_unset_key_and_empty_string_use_the_default_manager(monkeypatch):
    built = []

    class RecordingManager:
        def __init__(self, config, actor_wg, replicas):
            built.append(config.backend)

    monkeypatch.setattr(trainer_base, "CheckpointEngineManager", RecordingManager)
    _trainer()._build_checkpoint_manager()
    _trainer(checkpoint_manager_class="")._build_checkpoint_manager()
    assert built == ["naive", "naive"]
