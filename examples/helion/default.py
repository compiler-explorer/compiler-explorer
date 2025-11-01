import torch
import helion
import helion.language as hl


@helion.kernel(config=helion.Config(
    block_sizes=[],
    indexing='pointer',
    load_eviction_policies=['', ''],
    num_stages=2,
    num_warps=4,
    pid_type='flat',
    range_flattens=[None],
    range_multi_buffers=[None],
    range_num_stages=[0],
    range_unroll_factors=[0],
    range_warp_specializes=[]
), static_shapes=True)
def add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for idx in hl.grid(x.size()):
        out[idx] = (x[idx] + y[idx]) * scale
    return out


x = torch.rand(1024).cuda()
y = torch.rand(1024).cuda()
scale = 2.0
result = add(x, y, scale)
