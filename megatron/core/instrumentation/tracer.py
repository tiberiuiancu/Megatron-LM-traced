from __future__ import annotations

import json
import logging
import os
from time import perf_counter
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in this environment
    torch = None

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - codecarbon is optional
    EmissionsTracker = None


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
        self._total_flops: float | None = None
        self._energy_tracker: Any | None = None

    def set_total_flops(self, total_flops: float) -> None:
        self._total_flops = total_flops

    def _new_event(self):
        if not self._use_cuda_events:
            return None
        if len(self._event_pool) <= len(self._events):
            self._event_pool.append(torch.cuda.Event(enable_timing=True))
        return self._event_pool[len(self._events)]

    def start_iteration(self):
        self._events = []
        self._iteration_started = True
        self._energy_tracker = None
        if EmissionsTracker is not None:
            self._energy_tracker = EmissionsTracker(measure_power_secs=1)
            self._energy_tracker.start()
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = perf_counter()

    def record_collective(
        self,
        name,
        collective_type,
        bytes,
        group_ranks,
        microbatch_id=None,
        direction=None,
        async_op=False,
    ):
        event = self._new_event()
        metadata = {
            "name": name,
            "collective_type": collective_type,
            "bytes": bytes,
            "group_ranks": group_ranks,
            "microbatch_id": microbatch_id,
            "direction": direction,
            "async_op": async_op,
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
        trace: dict[str, Any] = {
            "trace_format_version": "1.0",
            "rank": self.rank,
            "world_size": self.world_size,
            "pipeline_stage": self.pipeline_stage,
            "events": [],
        }
        if self._total_flops is not None:
            trace["total_flops"] = self._total_flops

        if not self._iteration_started:
            return trace

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
        trace["events"] = events

        if self._energy_tracker is not None:
            try:
                self._energy_tracker.stop()
                energy_kwh = self._energy_tracker.final_emissions_data.energy_consumed
                co2eq_kg = self._energy_tracker.final_emissions
                trace["energy_kwh"] = energy_kwh
                trace["co2eq_kg"] = co2eq_kg
            except Exception:
                logging.warning("CodeCarbon energy tracking failed", exc_info=True)
            finally:
                self._energy_tracker = None

        return trace

    def save(self, trace_dir: str) -> None:
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"trace_rank_{self.rank}.json")
        trace = self.finish_iteration()
        serialize_trace(trace, path)


_tracer: CudaEventTracer | None = None


def get_tracer():
    return _tracer


def set_tracer(tracer):
    global _tracer
    _tracer = tracer
