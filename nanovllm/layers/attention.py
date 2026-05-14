import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        num_heads_per_kv = self.num_heads // self.num_kv_heads
        
        if context.is_prefill:
            outs = []
            # Gather scalar values back to CPU for sequence boundary slicing
            cu_seqlens_q = context.cu_seqlens_q.tolist()
            cu_seqlens_k = context.cu_seqlens_k.tolist()
            block_size = k_cache.size(1) if k_cache.numel() else 0
            
            for i in range(len(cu_seqlens_q) - 1):
                start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
                start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
                q_seq = q[start_q:end_q]
                
                if context.block_tables is not None:    # prefix cache active
                    seqlen_k = end_k - start_k
                    num_blocks = (seqlen_k + block_size - 1) // block_size
                    block_table = context.block_tables[i, :num_blocks].long()
                    
                    blocks_k = k_cache[block_table]
                    blocks_v = v_cache[block_table]
                    
                    # Flatten the relevant KV cache blocks and truncate padded tokens
                    k_seq = blocks_k.view(-1, self.num_kv_heads, self.head_dim)[:seqlen_k]
                    v_seq = blocks_v.view(-1, self.num_kv_heads, self.head_dim)[:seqlen_k]
                else:
                    k_seq = k[start_k:end_k]
                    v_seq = v[start_k:end_k]
                    
                q_i = q_seq.transpose(0, 1).unsqueeze(0)
                k_i = k_seq.transpose(0, 1).repeat_interleave(num_heads_per_kv, dim=0).unsqueeze(0)
                v_i = v_seq.transpose(0, 1).repeat_interleave(num_heads_per_kv, dim=0).unsqueeze(0)
                
                # PyTorch SDPA applies the causal mask bottom-right aligned automatically when causal=True
                out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True, scale=self.scale)
                outs.append(out_i.squeeze(0).transpose(0, 1))
                
            return torch.cat(outs, dim=0)
        else:
            bsz = q.size(0)
            q_b = q.unsqueeze(2)  # Target shape: (bsz, H_q, L_q=1, D)
            block_size = k_cache.size(1)
            
            # Use negative block clamp avoiding potential -1 fetches (completely CUDA graph safe)
            block_tables_safe = context.block_tables.clamp(min=0).long()
            blocks_k = k_cache[block_tables_safe]
            blocks_v = v_cache[block_tables_safe]
            
            max_seqlen = blocks_k.size(1) * blocks_k.size(2)
            k_seq = blocks_k.view(bsz, max_seqlen, self.num_kv_heads, self.head_dim)
            v_seq = blocks_v.view(bsz, max_seqlen, self.num_kv_heads, self.head_dim)
            
            # Repeat the KV heads to match Q heads for GQA (Grouped-Query Attention)
            k_b = k_seq.transpose(1, 2).repeat_interleave(num_heads_per_kv, dim=1)
            v_b = v_seq.transpose(1, 2).repeat_interleave(num_heads_per_kv, dim=1)
            
            # Create boolean mask handling padded tokens natively on the GPU (CUDA graph safe)
            idx = torch.arange(max_seqlen, device=q.device).unsqueeze(0)
            mask = idx < context.context_lens.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, max_seqlen)
            
            out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask, scale=self.scale)
            return out.transpose(1, 2)
