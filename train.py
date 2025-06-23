import unittest
from typing import List, Dict, Any
import random
from random import choices
import wandb

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass, asdict

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches

@dataclass
class TransformerConfig:
    """Configuration for Transformer model and training"""
    # Model parameters
    hidden_dim: int
    ff_dim: int
    num_heads: int
    num_layers: int
    vocab_size: int
    dropout_p: float = 0.1
    max_decoding_length: int = 10
    tie_output_to_embedding: bool = True

    # Training parameters
    n_epochs: int
    batch_size: int
    device: str
    
    # Optimizer parameters
    optimizer_name: str = "Adam"
    learning_rate: float = 0.0
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    
    # Scheduler parameters
    scheduler_name: str = "NoamOpt"
    warmup: int = 400
    factor: float = 1.0

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)

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
    config: TransformerConfig,
    run_name: str = "transformer_training",
):
    """
    Main training loop with wandb integration

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param config: the configuration for the transformer model and training
    :param run_name: name for the wandb run
    :return: the accuracy and loss on the latest batch
    """
    # Initialize wandb with minimal config
    config_dict = config.to_dict()
    config_dict["architecture"] = "Transformer"
    config_dict["device"] = config.device
    config_dict["optimizer"] = {
        "name": config.optimizer_name,
        "betas": config.betas,
        "eps": config.eps
    }
    config_dict["scheduler"] = {
        "name": config.scheduler_name,
        "warmup": config.warmup,
        "factor": config.factor
    }
    
    wandb.init(
        project="transformers_from_scratch",
        name=run_name,
        config=config_dict
    )

    transformer.train(True)
    num_iters = 0
    total_iters = config.n_epochs * len(batches["src"])
    epoch_losses = []
    epoch_accuracies = []

    print("\nTraining Progress:")
    print("-" * 60)
    print(f"{'Epoch':^6} | {'Iteration':^10} | {'Loss':^10} | {'Accuracy':^10} | {'Progress':^15}")
    print("-" * 60)

    for e in range(config.n_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            # Move batches to device
            src_batch = src_batch.to(config.device)
            src_mask = src_mask.to(config.device)
            tgt_batch = tgt_batch.to(config.device)
            tgt_mask = tgt_mask.to(config.device)

            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)

            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )

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

            # Log metrics to wandb
            wandb.log({
                "batch_loss": batch_loss.item(),
                "batch_accuracy": batch_accuracy.item(),
                "learning_rate": scheduler.optimizer.param_groups[0]["lr"],
                "epoch": e,
                "iteration": num_iters,
            })

            if num_iters % 5 == 0:
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

        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch_accuracy": avg_epoch_accuracy,
            "epoch": e,
        })

        print("-" * 60)
        print(f"Epoch {e} Summary - Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}")
        print("-" * 60)

    final_avg_loss = sum(epoch_losses) / len(epoch_losses)
    final_avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

    print("\nTraining Complete!")
    print(f"Final Average Loss: {final_avg_loss:.4f}")
    print(f"Final Average Accuracy: {final_avg_accuracy:.4f}")

    # Log final metrics and finish wandb run
    wandb.log({
        "final_loss": final_avg_loss,
        "final_accuracy": final_avg_accuracy,
    })
    wandb.finish()
    
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
        run_name = f"copy_task_{device.type}"
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
            config = TransformerConfig(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                vocab_size=en_vocab_size,
                n_epochs=10,
                batch_size=64,
                device=device.type,
                warmup=DEFAULT_CONFIG["warmup_steps"],
                factor=1.0
            )
        else:
            print("\nRunning with reduced model size on CPU")
            config = TransformerConfig(
                hidden_dim=DEFAULT_CONFIG["test_hidden_dim"],
                ff_dim=DEFAULT_CONFIG["test_ff_dim"],
                num_heads=DEFAULT_CONFIG["test_num_heads"],
                num_layers=DEFAULT_CONFIG["test_num_layers"],
                vocab_size=en_vocab_size,
                n_epochs=DEFAULT_CONFIG["test_n_epochs"],
                batch_size=DEFAULT_CONFIG["test_batch_size"],
                device=device.type,
                warmup=DEFAULT_CONFIG["test_warmup"],
                factor=2.0
            )
            print(f"Model config:")
            print(f"- Hidden dim: {config.hidden_dim}")
            print(f"- FF dim: {config.ff_dim}")
            print(f"- Attention heads: {config.num_heads}")
            print(f"- Layers: {config.num_layers}")
            print(f"- Epochs: {config.n_epochs}")

        transformer = Transformer(
            hidden_dim=config.hidden_dim,
            ff_dim=config.ff_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_decoding_length=config.max_decoding_length,
            vocab_size=config.vocab_size,
            padding_idx=en_vocab.token2index[en_vocab.PAD],
            bos_idx=en_vocab.token2index[en_vocab.BOS],
            dropout_p=config.dropout_p,
            tie_output_to_embedding=config.tie_output_to_embedding,
        ).to(device)

        # Test that the model can copy a sequence
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps
        )
        scheduler = NoamOpt(
            transformer.hidden_dim, 
            factor=config.factor, 
            warmup=config.warmup, 
            optimizer=optimizer,
        )

        # Create batches and masks
        batches, masks = construct_batches(
            corpus,
            en_vocab,
            batch_size=config.batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )
        
        print("\nStarting test training...")
        # Train
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, nn.CrossEntropyLoss(), batches, masks, config, run_name=run_name
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
    config = TransformerConfig(
        hidden_dim=DEFAULT_CONFIG["hidden_dim"],
        ff_dim=DEFAULT_CONFIG["ff_dim"],
        num_heads=DEFAULT_CONFIG["num_heads"],
        num_layers=DEFAULT_CONFIG["num_layers"],
        vocab_size=vocab_size,
        max_decoding_length=DEFAULT_CONFIG["max_decoding_length"],
        dropout_p=DEFAULT_CONFIG["dropout_p"],
        tie_output_to_embedding=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device.type,
        optimizer_name="Adam",
        learning_rate=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        scheduler_name="NoamOpt",
        warmup=DEFAULT_CONFIG["warmup_steps"],
        factor=1.0,
    )

    transformer = Transformer(
        hidden_dim=config.hidden_dim,
        ff_dim=config.ff_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_decoding_length=config.max_decoding_length,
        vocab_size=config.vocab_size,
        padding_idx=vocab.token2index[vocab.PAD],
        bos_idx=vocab.token2index[vocab.BOS],
        dropout_p=config.dropout_p,
        tie_output_to_embedding=config.tie_output_to_embedding,
    ).to(device)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps
    )
    scheduler = NoamOpt(
        transformer.hidden_dim, 
        factor=config.factor, 
        warmup=config.warmup, 
        optimizer=optimizer,
    )

    # 5. Loss Function
    criterion = nn.CrossEntropyLoss()

    # 6. Batch Construction
    batches, masks = construct_batches(
        corpus,
        vocab,
        batch_size=config.batch_size,
        src_lang_key="src",
        tgt_lang_key="tgt",
        device=device,
    )

    # 7. Training
    print("Starting training...")
    latest_batch_loss, latest_batch_accuracy = train(
        transformer, scheduler, criterion, batches, masks, config, run_name="transformer_training"
    )

    print(f"Training completed!")
    print(f"Final Batch Loss: {latest_batch_loss.item():.4f}")
    print(f"Final Batch Accuracy: {latest_batch_accuracy:.4f}")
