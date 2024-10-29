import random
from random import choices
from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches

import unittest

def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0

    for e in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )  # type: ignore

            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for.
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            # Rough estimate of per-token accuracy in the current training batch
            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / torch.numel(tgt_batch)

            if num_iters % 100 == 0:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )

            # Update parameters
            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # 1. Data Preparation
    synthetic_corpus_size = 600  # You can adjust this
    batch_size = 32
    n_epochs = 10
    n_tokens_in_batch = 10 

    corpus = ["These are the tokens that will end up in our vocabulary"]
    vocab = Vocabulary(corpus)
    vocab_size = len(list(vocab.token2index.keys()))
    valid_tokens = list(vocab.token2index.keys())[3:]
    corpus += [
        " ".join(choices(valid_tokens, k=n_tokens_in_batch))
        for _ in range(synthetic_corpus_size)
    ]
    corpus = [{"src": sent, "tgt": sent} for sent in corpus]

    # 2. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model Instantiation 
    transformer = Transformer(
        hidden_dim=256,  # Reduced
        ff_dim=1024,     # Reduced
        num_heads=4,       # Reduced
        num_layers=2,       # Reduced
        max_decoding_length=25,
        vocab_size=vocab_size,
        padding_idx=vocab.token2index[vocab.PAD],
        bos_idx=vocab.token2index[vocab.BOS],
        dropout_p=0.1,
        tie_output_to_embedding=True,
    ).to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamOpt(
        transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
    )

    # 5. Loss Function
    criterion = nn.CrossEntropyLoss()

    # 6. Batch Construction
    batches, masks = construct_batches(
        corpus,
        vocab,
        batch_size=batch_size,
        src_lang_key="src",
        tgt_lang_key="tgt",
        device=device,
    )

    # 7. Training
    latest_batch_loss, latest_batch_accuracy = train(
        transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
    )

    print(f"Final Batch Loss: {latest_batch_loss.item()}")
    print(f"Final Batch Accuracy: {latest_batch_accuracy}")