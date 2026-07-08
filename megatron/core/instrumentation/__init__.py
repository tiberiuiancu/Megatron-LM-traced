import torch

from .tracer import CudaEventTracer, get_tracer, set_tracer

__all__ = ["CudaEventTracer", "get_tracer", "set_tracer", "record_collective", "patch_torch_distributed"]


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


_patched = False
_orig = {}


def patch_torch_distributed():
    global _patched
    if _patched:
        return
    import torch.distributed as dist

    def _wrap(name, collective_type, fn, tensor_index):
        def wrapper(*args, **kwargs):
            group = kwargs.get("group")
            if group is None:
                for a in args:
                    if hasattr(a, "size") and hasattr(a, "rank"):
                        group = a
                        break
            tensor = args[tensor_index] if tensor_index < len(args) else None
            if tensor is not None and hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                record_collective(name, collective_type, tensor, group)
            return fn(*args, **kwargs)
        return wrapper

    _orig["all_gather"] = dist.all_gather
    _orig["reduce_scatter"] = dist.reduce_scatter
    _orig["all_reduce"] = dist.all_reduce
    _orig["all_to_all_single"] = dist.all_to_all_single

    dist.all_gather = _wrap("all_gather", "AllGather", dist.all_gather, 1)
    dist.reduce_scatter = _wrap("reduce_scatter", "ReduceScatter", dist.reduce_scatter, 1)
    dist.all_reduce = _wrap("all_reduce", "AllReduce", dist.all_reduce, 0)
    dist.all_to_all_single = _wrap("all_to_all_single", "AllToAll", dist.all_to_all_single, 1)

    for base_name, ct, ti in [
        ("_all_gather_base", "AllGather", 1),
        ("_reduce_scatter_base", "ReduceScatter", 1),
        ("broadcast", "Broadcast", 0),
    ]:
        fn = getattr(dist, base_name, None)
        if fn is not None:
            _orig[base_name] = fn
            setattr(dist, base_name, _wrap(base_name, ct, fn, ti))

    _patched = True
