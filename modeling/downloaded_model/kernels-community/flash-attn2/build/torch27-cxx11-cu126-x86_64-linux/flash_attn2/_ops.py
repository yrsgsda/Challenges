import torch
from . import _flash_attn_9e27194
ops = torch.ops._flash_attn_9e27194

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_flash_attn_9e27194::{op_name}"