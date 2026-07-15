"""Shared model/tokenizer setup for semantic-ID training and eval."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .semid import SidTable


def prepare(base: str, table: SidTable, dtype=torch.bfloat16):
    """Load base model + tokenizer, add sid tokens, mean-init their embeddings.

    Returns (tokenizer, model, new_token_ids).
    """
    tok = AutoTokenizer.from_pretrained(base)
    added = tok.add_tokens(table.tokens(), special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if len(tok) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tok))
    new_ids = [tok.convert_tokens_to_ids(t) for t in table.tokens()]
    if added:
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean = emb[: len(tok) - added].mean(0)
            for i in new_ids:
                emb[i] = mean + 0.02 * torch.randn_like(mean)
    return tok, model, new_ids
