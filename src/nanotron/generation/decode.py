import dataclasses
import time
from itertools import chain, islice
from typing import TYPE_CHECKING, Generator, Iterable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import BenchArgs, GenerationArgs
from nanotron.generation.sampler import BasicSampler, GreedySampler, SamplerType, TopKSampler, TopPSampler
from nanotron.helpers import log_throughput
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext

if TYPE_CHECKING:
    try:
        from transformers import PreTrainedTokenizer
    except ImportError:
        PreTrainedTokenizer = None


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class GenerationInput:
    text: str


@dataclasses.dataclass
class GenerationInputs:
    input_ids: torch.Tensor  # [B, S]
    input_masks: torch.Tensor


@dataclasses.dataclass
class GenerationOutput:
    input_ids: torch.Tensor
    generation_ids: torch.Tensor
    return_logits: Optional[torch.Tensor] = None


@dataclasses.dataclass
class TokenizerConfig:
    max_input_length: Optional[int]
    truncation: Optional[Union[str, bool]] = None
    padding: Optional[Union[str, bool]] = None


def chunks(iterable, chunk_size: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from `iterable`"""
    assert chunk_size >= 1
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, chunk_size - 1)))


def micro_batcher(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    max_micro_batch_size: int,
    tokenizer_config: TokenizerConfig,
    parallel_context: ParallelContext,
    input_rank: int = 0,
) -> Generator[GenerationInputs, None, None]:
    """
    Returns:
        input_ids: [max_micro_batch_size, max_input_length]
        input_masks: [max_micro_batch_size, max_input_length]
    """
    if tokenizer_config.padding is None:
        tokenizer_config.padding = "max_length" if tokenizer_config.max_input_length is not None else True
    if tokenizer_config.truncation is None:
        tokenizer_config.truncation = True if tokenizer_config.max_input_length is not None else None

    # Assuming DP=1
    for micro_batch in chunks(input_iter, chunk_size=max_micro_batch_size):
        if len(micro_batch) == 0:
            return

        encodings = tokenizer(
            [elt.text for elt in micro_batch],
            return_tensors="pt",
            return_attention_mask=True,
            padding=tokenizer_config.padding,
            max_length=tokenizer_config.max_input_length,
            truncation=tokenizer_config.truncation,
        )

        encodings["attention_mask"] = encodings.attention_mask.to(dtype=torch.bool, device="cuda")
        # encodings.to("cuda") # input_ids to cuda?
        input_ids = encodings.input_ids.to("cuda")
        
        yield GenerationInputs(input_ids=input_ids, input_masks=encodings.attention_mask)


@torch.inference_mode()
def get_position_ids(input_ids, tokenizer):
    # Find where padding ends for each sequence
    batch_size, seq_length = input_ids.shape
    padding_token_id = tokenizer.eos_token_id

    # Create a mask of padding tokens
    padding_mask = input_ids == padding_token_id

    # Find indices where non-padding tokens start
    non_padding_indices = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    # Only compute for rows that have padding
    rows_with_padding = padding_mask.any(dim=1).nonzero().squeeze(-1)
    if rows_with_padding.numel() > 0:
        for idx in rows_with_padding:
            # Find the last padding token position
            last_padding_pos = padding_mask[idx].nonzero()[-1].item()
            non_padding_indices[idx] = last_padding_pos + 1

    # Create position_ids tensor initialized with zeros
    position_ids = torch.zeros((batch_size, seq_length), dtype=torch.int32, device=input_ids.device)

    # Set up a range tensor we'll use for all sequences
    range_tensor = torch.arange(seq_length, device=input_ids.device)

    # Vectorized assignment of position IDs
    for i in range(batch_size):
        start_idx = non_padding_indices[i].item()
        # For positions after padding, use range starting from 0
        # For positions before or at padding end, keep as 0
        if start_idx < seq_length:
            position_ids[i, start_idx:] = range_tensor[: seq_length - start_idx]
    return position_ids


@torch.inference_mode()
def decode_text(
    input_iter: Iterable[GenerationInput],
    tokenizer: "PreTrainedTokenizer",
    model: NanotronModel,
    parallel_context: ParallelContext,
    generation_config: GenerationArgs,
    tokenizer_config: Optional[TokenizerConfig],
    max_micro_batch_size: int,
    max_new_tokens: int,
    is_bench: bool = False,
    logits_are_batch_first: bool = True,
) -> Generator[GenerationOutput, None, None]:
    
    if generation_config:
        if isinstance(generation_config.sampler, str):
            sampler_type = SamplerType(generation_config.sampler.upper())
        else:
            sampler_type = generation_config.sampler
    else:
        sampler_type = SamplerType.GREEDY

    # Sampler init
    if sampler_type == SamplerType.GREEDY:
        sampler = GreedySampler(pg=parallel_context.tp_pg)
    elif sampler_type == SamplerType.TOP_K:
        sampler = TopKSampler(
            pg=parallel_context.tp_pg,
            k=generation_config.top_k,
            temperature=generation_config.temperature,
        )
    elif sampler_type == SamplerType.TOP_P:
        sampler = TopPSampler(
            pg=parallel_context.tp_pg,
            p=generation_config.top_p,
            temperature=generation_config.temperature,
        )
    elif sampler_type == SamplerType.BASIC:
        sampler = BasicSampler(pg=parallel_context.tp_pg)
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} is not implemented")

    # replicate input for n_samples
    if generation_config and generation_config.n_samples:
        if sampler_type != SamplerType.TOP_P and sampler_type != SamplerType.TOP_K:
            raise ValueError("Only support n_samples for TOP_P and TOP_K sampler")
        input_iter = [
            GenerationInput(text=input.text) for input in input_iter for _ in range(generation_config.n_samples)
        ]

    for batch in micro_batcher(
        input_iter=input_iter,
        tokenizer=tokenizer,
        max_micro_batch_size=max_micro_batch_size,
        tokenizer_config=tokenizer_config,
        parallel_context=parallel_context,
    ):
        input_ids = batch.input_ids
        input_masks = batch.input_masks
        
        current_ids = input_ids
        generated_ids = []
        
        if is_bench:
            start_time = time.perf_counter()

        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            position_ids = get_position_ids(current_ids, tokenizer)
            
            if hasattr(model, "model") and hasattr(model.model, "forward"):
                 # It is likely Qwen2ForTraining or similar wrapper
                 sharded_logits = model.model(input_ids=current_ids, position_ids=position_ids)
            else:
                 sharded_logits = model(input_ids=current_ids, position_ids=position_ids)
            
            if isinstance(sharded_logits, dict):
                sharded_logits = sharded_logits.get("logits", sharded_logits.get("sharded_logits"))
            
            # Reshape
            sharded_logits = sharded_logits.view(current_ids.shape[0], current_ids.shape[1], -1)
            
            if not logits_are_batch_first:
                sharded_logits = sharded_logits.transpose(0, 1)
                
            next_token_logits = sharded_logits[:, -1, :]
            next_token = sampler(sharded_logits=next_token_logits)
            
            generated_ids.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
        # Yield output
        generated_ids = torch.cat(generated_ids, dim=-1)
        
        # Batch yield
        for i in range(len(input_ids)):
            yield GenerationOutput(
                input_ids=input_ids[i],
                generation_ids=generated_ids[i]
            )

def decode_tokenized(*args, **kwargs):
    raise NotImplementedError("decode_tokenized not implemented in simplified version")
