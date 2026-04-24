Complete the assignment 05 of the CS336 class, your goal is to build supervised fine-tuning (SFT), direct preference optimization (DPO), and group-relative policy optimization (GRPO) training loops plus the shared data/metrics plumbing from scratch.

What you can use We expect you to build these components from scratch. In particular, you may not use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
• torch.nn.Parameter
• Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)
• The torch.optim.Optimizer base class

You may use transformers, vllm, and flash_attn for model loading, generation, and attention — these are installed.

Do not stop until all your tests pass.
