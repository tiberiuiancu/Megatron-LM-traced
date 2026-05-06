from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in this environment
    torch = None


def serialize_trace(trace: dict, path: str) -> None:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trace, f, indent=2)
    os.replace(tmp, path)


class CudaEventTracer:
    def __init__(self, rank: int = 0, world_size: int = 0, pipeline_stage: int = 0):
        self.rank = rank
        self.world_size = world_size
        self.pipeline_stage = pipeline_stage
        self._iteration_started = False
        self._use_cuda_events = bool(torch is not None and torch.cuda.is_available())
        self._start_time = 0.0
        self._start_event = None
        self._events: list[tuple[str, float | Any, dict[str, Any]]] = []
        self._event_pool: list[Any] = []

    def _new_event(self):
        if not self._use_cuda_events:
            return None
        if len(self._event_pool) <= len(self._events):
            self._event_pool.append(torch.cuda.Event(enable_timing=True))
        return self._event_pool[len(self._events)]

    def start_iteration(self):
        self._events = []
        self._iteration_started = True
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = perf_counter()

    def record_collective(self, name, collective_type, bytes, group_ranks):
        event = self._new_event()
        metadata = {
            "name": name,
            "collective_type": collective_type,
            "bytes": bytes,
            "group_ranks": group_ranks,
        }
        if self._use_cuda_events:
            event.record()
            self._events.append(("collective", event, metadata))
        else:
            self._events.append(("collective", perf_counter(), metadata))

    def record_slot_begin(self, microbatch_id, direction, pipeline_stage):
        self._record_slot_event("slot_begin", microbatch_id, direction, pipeline_stage)

    def record_slot_end(self, microbatch_id, direction, pipeline_stage):
        self._record_slot_event("slot_end", microbatch_id, direction, pipeline_stage)

    def _record_slot_event(self, event_type, microbatch_id, direction, pipeline_stage):
        event = self._new_event()
        metadata = {
            "microbatch_id": microbatch_id,
            "direction": direction,
            "pipeline_stage": pipeline_stage,
        }
        if self._use_cuda_events:
            event.record()
            self._events.append((event_type, event, metadata))
        else:
            self._events.append((event_type, perf_counter(), metadata))

    def finish_iteration(self) -> dict:
        if not self._iteration_started:
            return {
                "trace_format_version": "1.0",
                "rank": self.rank,
                "world_size": self.world_size,
                "pipeline_stage": self.pipeline_stage,
                "events": [],
            }

        if self._use_cuda_events:
            torch.cuda.synchronize()
            events = [
                {
                    "type": event_type,
                    "timestamp_ms": float(self._start_event.elapsed_time(event)),
                    "metadata": metadata,
                }
                for event_type, event, metadata in self._events
            ]
        else:
            events = [
                {
                    "type": event_type,
                    "timestamp_ms": float((timestamp - self._start_time) * 1000.0),
                    "metadata": metadata,
                }
                for event_type, timestamp, metadata in self._events
            ]

        events.sort(key=lambda item: item["timestamp_ms"])
        self._iteration_started = False
        return {
            "trace_format_version": "1.0",
            "rank": self.rank,
            "world_size": self.world_size,
            "pipeline_stage": self.pipeline_stage,
            "events": events,
        }


_tracer: CudaEventTracer | None = None


def get_tracer():
    return _tracer


def set_tracer(tracer):
    global _tracer
    _tracer = tracer
