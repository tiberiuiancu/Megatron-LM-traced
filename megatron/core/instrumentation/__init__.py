import torch

from .tracer import CudaEventTracer, get_tracer, set_tracer

__all__ = ["CudaEventTracer", "get_tracer", "set_tracer", "record_collective"]


def _group_ranks(group):
    if hasattr(group, "ranks"):
        ranks = group.ranks
        return list(ranks() if callable(ranks) else ranks)
    get_process_group_ranks = getattr(torch.distributed, "get_process_group_ranks", None)
    if get_process_group_ranks is not None:
        try:
            return list(get_process_group_ranks(group))
        except Exception:
            pass
    return list(range(torch.distributed.get_world_size(group)))


def record_collective(name, collective_type, tensor, group):
    tracer = get_tracer()
    if tracer is not None:
        tracer.record_collective(
            name=name,
            collective_type=collective_type,
            bytes=tensor.numel() * tensor.element_size(),
            group_ranks=_group_ranks(group),
        )
