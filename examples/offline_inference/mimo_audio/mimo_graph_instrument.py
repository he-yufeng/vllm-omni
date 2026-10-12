# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation hooks for MiMo-Audio's private CUDA graphs.

Two independent switches, both no-ops unless the env var is set:

- ``MIMO_GRAPH_INSTRUMENT=1``: wrap both capture entrypoints, record which
  batch-size buckets were captured, count every replay per bucket, and dump a
  JSON report at exit. A bucket that captured but never replayed shows up with
  a zero count; a silent eager fallback shows up as an empty bucket list (plus
  the startup warning from the model constructor).
- ``MIMO_GRAPH_FORCE_EAGER=1``: make every capture raise, so the constructor
  falls back to the eager path. This is the controlled off-switch for an eager
  reference run; ``enforce_eager=True`` does not cover these graphs because
  capture is gated only on ``torch.cuda.is_available()``.

Enable from end2end.py (done automatically when either env var is set), or
``import mimo_graph_instrument; mimo_graph_instrument.install()`` before
constructing ``Omni(...)``.
"""

import atexit
import json
import os
from collections import Counter

_installed = False

_captured: dict[str, list[int]] = {"local_decode": [], "input_local": []}
_replays: Counter[str] = Counter()


def _wrap_capture(cls, family: str) -> None:
    original = cls.capture.__func__

    @classmethod
    def wrapped(cls_, *args, **kwargs):
        inst = original(cls_, *args, **kwargs)
        _captured[family].append(inst.batch_size)
        real_replay = inst.cuda_graph.replay

        def counted_replay():
            _replays[f"{family}:bs{inst.batch_size}"] += 1
            return real_replay()

        inst.cuda_graph.replay = counted_replay
        return inst

    cls.capture = wrapped


def _force_eager(cls, family: str) -> None:
    @classmethod
    def raising(cls_, *args, **kwargs):
        raise RuntimeError(f"MIMO_GRAPH_FORCE_EAGER: {family} capture disabled for eager reference run")

    cls.capture = raising


def _report() -> None:
    print(
        "CAPTURE/REPLAY REPORT "
        + json.dumps({"captured_buckets": _captured, "replay_counts": dict(_replays)}, sort_keys=True)
    )


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    from vllm_omni.model_executor.models.mimo_audio import mimo_audio_llm

    classes = [
        (mimo_audio_llm.MiMoLocalDecodeCudaGraph, "local_decode"),
        (mimo_audio_llm.MiMoInputLocalTransformerCudaGraph, "input_local"),
    ]
    if os.environ.get("MIMO_GRAPH_FORCE_EAGER"):
        for cls, family in classes:
            _force_eager(cls, family)
    if os.environ.get("MIMO_GRAPH_INSTRUMENT"):
        for cls, family in classes:
            _wrap_capture(cls, family)
        atexit.register(_report)
