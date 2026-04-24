Complete the assignment 02 of the CS336 class, your goal is to build FlashAttention-2 (forward + backward in Triton), a distributed data-parallel wrapper, and a ZeRO-1 sharded optimizer from scratch.

What you can use We expect you to build these components from scratch. In particular, you may not use any definitions from torch.nn, torch.nn.functional, torch.optim, or torch.nn.parallel except for the following:
• torch.nn.Parameter
• Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)
• The torch.optim.Optimizer base class
• torch.distributed collectives (all_reduce, all_gather, broadcast, reduce_scatter)
• Triton

Do not stop until all your tests pass.
