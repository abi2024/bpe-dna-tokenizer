"""
BPE DNA Tokenizer Package

A Byte Pair Encoding tokenizer implementation for DNA sequences.
"""

from .bpe_tokenizer import (
    BPETokenizer,
    get_pair_counts,
    merge_pair,
    initialize_vocab,
    train_bpe,
    encode,
    decode,
)

from .data_loader import (
    load_fasta,
    download_ecoli_genome,
    get_sequence_stats,
)

from .utils import (
    calculate_compression_ratio,
    save_tokenizer,
    load_tokenizer,
    analyze_vocabulary,
    find_biological_patterns,
    plot_results,
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # BPE Core
    "BPETokenizer",
    "get_pair_counts",
    "merge_pair",
    "initialize_vocab",
    "train_bpe",
    "encode",
    "decode",
    
    # Data Loading
    "load_fasta",
    "download_ecoli_genome",
    "get_sequence_stats",
    
    # Utilities
    "calculate_compression_ratio",
    "save_tokenizer",
    "load_tokenizer",
    "analyze_vocabulary",
    "find_biological_patterns",
    "plot_results",
]