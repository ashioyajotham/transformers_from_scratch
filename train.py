import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches

# Configuration parameters
DEFAULT_CONFIG = {
    "synthetic_corpus_size": 600,
    "batch_size": 32,
    "n_epochs": 10,
    "n_tokens_in_batch": 10,
    "hidden_dim": 256,
    "ff_dim": 1024,
    "num_heads": 4,
    "num_layers": 2,
    "max_decoding_length": 25,
    "dropout_p": 0.1,
    "warmup_steps": 400,
    # Test specific configurations (smaller for CPU)
    "test_hidden_dim": 32,
    "test_ff_dim": 128,
    "test_num_heads": 2,
    "test_num_layers": 1,
    "test_batch_size": 2,
    "test_n_epochs": 20,
    "test_warmup": 100,
}

def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
    device: torch.device,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :param device: device to train on (cuda/cpu)
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0
    total_iters = n_epochs * len(batches["src"])
    epoch_losses = []
    epoch_accuracies = []

    print("\nTraining Progress:")
    print("-" * 60)
    print(f"{'Epoch':^6} | {'Iteration':^10} | {'Loss':^10} | {'Accuracy':^10} | {'Progress':^15}")
    print("-" * 60)

    for e in range(n_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            # Move batches to device
            src_batch = src_batch.to(device)
            src_mask = src_mask.to(device)
            tgt_batch = tgt_batch.to(device)
            tgt_mask = tgt_mask.to(device)

            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )  # type: ignore

            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]

            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / torch.numel(tgt_batch)

            # Update running averages
            epoch_loss += batch_loss.item()
            epoch_accuracy += batch_accuracy.item()
            num_batches += 1

            if num_iters % 5 == 0:  # Print every 5 iterations
                progress = f"{num_iters}/{total_iters}"
                print(f"{e:^6} | {num_iters:^10} | {batch_loss.item():^10.4f} | {batch_accuracy.item():^10.4f} | {progress:^15}")

            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1

        # Compute epoch averages
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        epoch_losses.append(avg_epoch_loss)
        epoch_accuracies.append(avg_epoch_accuracy)

        print("-" * 60)
        print(f"Epoch {e} Summary - Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}")
        print("-" * 60)

    print("\nTraining Complete!")
    print(f"Final Average Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")
    print(f"Final Average Accuracy: {sum(epoch_accuracies) / len(epoch_accuracies):.4f}")
    
    return batch_loss, batch_accuracy


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def test_copy_task(self):
        """
        Test training by trying to (over)fit a simple copy dataset.
        Will run on GPU if available, otherwise on CPU with reduced model size.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning test on: {device}")

        # Create (shared) vocabulary and special token indices given a dummy corpus
        corpus = [
            "Hello world",  
            "Testing copy",
        ]
        print(f"\nTest corpus: {corpus}")
        print("Tokenized corpus:")
        
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        print(f"Vocabulary size: {en_vocab_size}")
        print("Vocabulary:", list(en_vocab.token2index.keys()))

        # Adjust model size based on device
        if device.type == "cuda":
            hidden_dim, ff_dim = 512, 2048
            num_heads, num_layers = 8, 6
            n_epochs = 10
        else:
            print("\nRunning with reduced model size on CPU")
            hidden_dim = DEFAULT_CONFIG["test_hidden_dim"]
            ff_dim = DEFAULT_CONFIG["test_ff_dim"]
            num_heads = DEFAULT_CONFIG["test_num_heads"]
            num_layers = DEFAULT_CONFIG["test_num_layers"]
            n_epochs = DEFAULT_CONFIG["test_n_epochs"]
            print(f"Model config:")
            print(f"- Hidden dim: {hidden_dim}")
            print(f"- FF dim: {ff_dim}")
            print(f"- Attention heads: {num_heads}")
            print(f"- Layers: {num_layers}")
            print(f"- Epochs: {n_epochs}")

        transformer = Transformer(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_decoding_length=10,
            vocab_size=en_vocab_size,
            padding_idx=en_vocab.token2index[en_vocab.PAD],
            bos_idx=en_vocab.token2index[en_vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Test that the model can copy a sequence
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        if device.type == "cuda":
            scheduler = NoamOpt(
                transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer
            )
        else:
            scheduler = NoamOpt(
                transformer.hidden_dim, factor=2.0, warmup=DEFAULT_CONFIG["test_warmup"], optimizer=optimizer
            )

        # Create batches and masks
        batches, masks = construct_batches(
            corpus,
            en_vocab,
            batch_size=DEFAULT_CONFIG["test_batch_size"],
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )
        
        print("\nStarting test training...")
        # Train
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, nn.CrossEntropyLoss(), batches, masks, 
            n_epochs=n_epochs, device=device
        )
        
        # For CPU, we'll be more lenient with the success criteria
        if device.type == "cuda":
            self.assertTrue(latest_batch_loss.item() <= 0.01)
            self.assertTrue(latest_batch_accuracy >= 0.99)
        else:
            print(f"\nFinal test results - Loss: {latest_batch_loss.item():.4f}, Accuracy: {latest_batch_accuracy:.4f}")
            self.assertTrue(latest_batch_loss.item() <= 0.5)  
            self.assertTrue(latest_batch_accuracy >= 0.8)  


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    synthetic_corpus_size = DEFAULT_CONFIG["synthetic_corpus_size"]
    batch_size = DEFAULT_CONFIG["batch_size"]
    n_epochs = DEFAULT_CONFIG["n_epochs"]
    n_tokens_in_batch = DEFAULT_CONFIG["n_tokens_in_batch"]

    corpus = ["These are the tokens that will end up in our vocabulary"]
    vocab = Vocabulary(corpus)
    vocab_size = len(list(vocab.token2index.keys()))
    valid_tokens = list(vocab.token2index.keys())[3:]  # Skip special tokens
    corpus += [
        " ".join(choices(valid_tokens, k=n_tokens_in_batch))
        for _ in range(synthetic_corpus_size)
    ]
    corpus = [{"src": sent, "tgt": sent} for sent in corpus]

    # 3. Model Instantiation 
    transformer = Transformer(
        hidden_dim=DEFAULT_CONFIG["hidden_dim"],
        ff_dim=DEFAULT_CONFIG["ff_dim"],
        num_heads=DEFAULT_CONFIG["num_heads"],
        num_layers=DEFAULT_CONFIG["num_layers"],
        max_decoding_length=DEFAULT_CONFIG["max_decoding_length"],
        vocab_size=vocab_size,
        padding_idx=vocab.token2index[vocab.PAD],
        bos_idx=vocab.token2index[vocab.BOS],
        dropout_p=DEFAULT_CONFIG["dropout_p"],
        tie_output_to_embedding=True,
    ).to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamOpt(
        transformer.hidden_dim, 
        factor=1, 
        warmup=DEFAULT_CONFIG["warmup_steps"], 
        optimizer=optimizer,
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
    print("Starting training...")
    latest_batch_loss, latest_batch_accuracy = train(
        transformer, scheduler, criterion, batches, masks, 
        n_epochs=n_epochs, device=device
    )

    print(f"Training completed!")
    print(f"Final Batch Loss: {latest_batch_loss.item():.4f}")
    print(f"Final Batch Accuracy: {latest_batch_accuracy:.4f}")
