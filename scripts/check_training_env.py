"""Check the Megatron/TE binary stack; optionally execute a CUDA backward pass."""
import argparse
import importlib.metadata as metadata
import json
import os


def check(require_gpu=False):
    import torch
    import transformer_engine.pytorch as te
    import fused_weight_gradient_mlp_cuda
    from megatron.core.tensor_parallel.layers import _grad_accum_fusion_available
    if not _grad_accum_fusion_available:
        raise RuntimeError("Megatron cannot load the Apex gradient accumulation kernel")
    from megatron.core.models.common.embeddings.rope_utils import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )
    from verl.models.mcore.mbridge import AutoBridge
    from mbridge.core.parallel_states import ParallelStates
    from transformers import Qwen3MoeConfig

    versions = {name: metadata.version(name) for name in (
        'torch', 'vllm', 'ray', 'megatron-core', 'mbridge',
        'transformer-engine', 'transformer-engine-torch', 'apex', 'flash-attn', 'transformers',
    )}
    print(json.dumps(versions, indent=2), flush=True)
    if fused_apply_rotary_pos_emb is None and fused_apply_rotary_pos_emb_thd is None:
        raise RuntimeError('Megatron cannot load the Transformer Engine RoPE kernels')
    hf_config = Qwen3MoeConfig(num_hidden_layers=2, hidden_size=256,
        intermediate_size=512, moe_intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4, head_dim=64,
        num_experts=8, num_experts_per_tok=2, vocab_size=1024)
    bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16,
        parallel_states=ParallelStates())
    bridge.set_extra_args(apply_rope_fusion=True, moe_router_dtype='fp32',
        moe_permute_fusion=True, gradient_accumulation_fusion=True)
    assert bridge.config.apply_rope_fusion
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this check')
        layer = te.Linear(256, 256, params_dtype=torch.bfloat16, device='cuda')
        x = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y = layer(x)
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grad_output = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16)
        main_grad = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(x.detach(), grad_output, main_grad)
        torch.cuda.synchronize()
        expected = grad_output.float().T @ x.detach().float()
        torch.testing.assert_close(main_grad, expected, rtol=0.02, atol=0.02)
        os.environ['NVTE_FLASH_ATTN'] = '1'
        os.environ['NVTE_FUSED_ATTN'] = '0'
        os.environ['NVTE_UNFUSED_ATTN'] = '0'
        attention = te.DotProductAttention(num_attention_heads=4, kv_channels=64,
            attention_dropout=0.0, attn_mask_type='causal', qkv_format='sbhd')
        q, k, v = [torch.randn(8, 1, 4, 64, device='cuda', dtype=torch.bfloat16,
            requires_grad=True) for _ in range(3)]
        result = attention(q, k, v)
        result.float().square().mean().backward()
        torch.cuda.synchronize()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        print('TE forward/backward, FlashAttention and Apex fused weight gradient passed', flush=True)
    print('Megatron/MBridge configuration with fused RoPE passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    check(parser.parse_args().require_gpu)
