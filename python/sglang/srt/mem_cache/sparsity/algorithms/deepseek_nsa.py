from __future__ import annotations
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

from klogger import IndexLogConfig, TopKLogger


class DeepSeekNSAAlgorithm(BaseSparseAlgorithmImpl):
    """
    Sparse attention algorithm for DeepSeek NSA.

    This algorithm uses NSA's native indexer for TopK retrieval.
    Overrides all parent methods as NSA has its own specialized flow.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)

        # Default logger config, no overrides.
        self._topk_log_cfg = IndexLogConfig()
        self._topk_logger = TopKLogger(self._topk_log_cfg)

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> tuple:
        indexer, forward_batch, x, q_lora, positions = (
            kwargs.get("indexer"),
            kwargs.get("forward_batch"),
            kwargs.get("x"),
            kwargs.get("q_lora"),
            kwargs.get("positions"),
        )

        if any(v is None for v in [indexer, x, q_lora, positions, forward_batch]):
            raise ValueError("Required: indexer, forward_batch, x, q_lora, positions")

        topk = indexer(
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
        )

        run_id = 0

        # If topk is None -> enqueue a sentinel tensor of -2s
        if topk is None:
            device = queries.device if torch.is_tensor(queries) else "cuda"
            topk_to_log = torch.full((1, 1), -2, dtype=torch.int32, device=device)
            self._topk_logger.enqueue(
                topk_to_log, run_id=run_id, layer_id=layer_id, start_pos=0
            )
            return (topk, None)

        try:
            if torch.is_tensor(topk) and topk.numel() > 0 and (topk == -1).all().item():
                return (topk, None)
        except Exception:
            pass

        self._topk_logger.enqueue(topk, run_id=run_id, layer_id=layer_id, start_pos=0)
        return (topk, None)

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        pass

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ):
        pass

    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ):
        pass
