"""
Core BPE Tokenizer Implementation

This module contains the main BPE training and inference functions.
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional
import time


def initialize_vocab(corpus: str) -> Dict[str, int]:
    """
    Create initial vocabulary from unique characters in corpus.
    
    Args:
        corpus: DNA sequence string
        
    Returns:
        Dictionary mapping characters to IDs
        
    Example:
        >>> initialize_vocab("ATCG")
        {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    """
    unique_chars = sorted(set(corpus))
    vocab = {char: idx for idx, char in enumerate(unique_chars)}
    return vocab


def get_pair_counts(tokens: List[str]) -> Counter:
    """
    Count frequency of all adjacent token pairs.
    
    Args:
        tokens: List of current tokens (strings)
        
    Returns:
        Counter object with pair frequencies
        
    Example:
        >>> get_pair_counts(['A', 'T', 'A', 'T'])
        Counter({('A', 'T'): 2, ('T', 'A'): 1})
    """
    pairs = []
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs.append(pair)
    
    return Counter(pairs)


def merge_pair(tokens: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
    """
    Replace all occurrences of pair with new_token.
    
    Args:
        tokens: List of current tokens
        pair: Tuple (token1, token2) to merge
        new_token: String to replace pair with
        
    Returns:
        New list of tokens with pairs merged
        
    Example:
        >>> merge_pair(['A', 'T', 'C', 'A', 'T'], ('A', 'T'), 'AT')
        ['AT', 'C', 'AT']
    """
    new_tokens = []
    i = 0
    
    while i < len(tokens):
        # Check if we found the pair
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(new_token)
            i += 2  # Skip both tokens in the pair
        else:
            new_tokens.append(tokens[i])
            i += 1
    
    return new_tokens


def train_bpe(corpus: str, 
              target_vocab_size: int,
              verbose: bool = True,
              progress_interval: int = 100) -> Tuple[Dict[str, int], List[Tuple[str, str]], List[str]]:
    """
    Train BPE tokenizer on corpus.
    
    Args:
        corpus: DNA sequence string
        target_vocab_size: Desired vocabulary size
        verbose: Whether to print progress
        progress_interval: Print progress every N merges
        
    Returns:
        Tuple of (vocab, merge_rules, final_tokens)
        
    Example:
        >>> vocab, rules, tokens = train_bpe("ATCGATCG", target_vocab_size=10)
    """
    if verbose:
        print("Starting BPE training...")
        print(f"Corpus size: {len(corpus):,} bases")
        print(f"Target vocab size: {target_vocab_size:,}")
    
    start_time = time.time()
    
    # Initialize
    tokens = list(corpus)
    vocab = initialize_vocab(corpus)
    initial_vocab_size = len(vocab)
    
    if verbose:
        print(f"Initial vocab size: {initial_vocab_size}")
    
    num_merges = target_vocab_size - initial_vocab_size
    
    if num_merges <= 0:
        raise ValueError(f"Target vocab size ({target_vocab_size}) must be greater than initial vocab size ({initial_vocab_size})")
    
    if verbose:
        print(f"Number of merges needed: {num_merges:,}\n")
    
    merge_rules = []
    
    # Main training loop
    if verbose:
        print("Training progress:")
    
    for i in range(num_merges):
        # Count pairs
        pair_counts = get_pair_counts(tokens)
        
        if not pair_counts:
            if verbose:
                print(f"\nNo more pairs to merge at iteration {i}")
            break
        
        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]
        
        # Create new token
        new_token = best_pair[0] + best_pair[1]
        
        # Add to vocabulary
        vocab[new_token] = len(vocab)
        
        # Record merge rule
        merge_rules.append(best_pair)
        
        # Apply merge
        tokens = merge_pair(tokens, best_pair, new_token)
        
        # Progress reporting
        if verbose and (i + 1) % progress_interval == 0:
            compression = len(corpus) / len(tokens)
            print(f"  Merge {i+1:>4}/{num_merges}: "
                  f"Merged {best_pair} ({best_count:,} times) -> '{new_token}' | "
                  f"Tokens: {len(tokens):,} | "
                  f"Compression: {compression:.2f}x")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Training complete in {elapsed:.2f} seconds")
        print(f"Final vocab size: {len(vocab):,}")
        print(f"Final token count: {len(tokens):,}")
    
    return vocab, merge_rules, tokens


def encode(text: str, 
           vocab: Dict[str, int], 
           merge_rules: List[Tuple[str, str]]) -> List[int]:
    """
    Encode text using trained tokenizer.
    
    Args:
        text: DNA sequence string to encode
        vocab: Trained vocabulary
        merge_rules: List of merge rules from training
        
    Returns:
        List of token IDs
        
    Example:
        >>> encode("ATCG", vocab, merge_rules)
        [42, 15, 23]
    """
    # Start with character-level tokenization
    tokens = list(text)
    
    # Apply merge rules in order
    for pair in merge_rules:
        merged_token = pair[0] + pair[1]
        tokens = merge_pair(tokens, pair, merged_token)
    
    # Convert to IDs
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            # Unknown token - should rarely happen
            print(f"Warning: Unknown token '{token}' encountered during encoding")
    
    return token_ids


def decode(token_ids: List[int], vocab: Dict[str, int]) -> str:
    """
    Decode token IDs back to original text.
    
    Args:
        token_ids: List of integer token IDs
        vocab: Trained vocabulary
        
    Returns:
        Decoded DNA sequence string
        
    Example:
        >>> decode([42, 15, 23], vocab)
        "ATCG"
    """
    # Create reverse vocabulary
    id_to_token = {idx: token for token, idx in vocab.items()}
    
    # Look up each ID
    tokens = []
    for token_id in token_ids:
        if token_id in id_to_token:
            tokens.append(id_to_token[token_id])
        else:
            print(f"Warning: Unknown token ID {token_id} encountered during decoding")
    
    # Concatenate
    decoded_text = ''.join(tokens)
    
    return decoded_text


class BPETokenizer:
    """
    Object-oriented wrapper for BPE tokenizer.
    
    This class provides a scikit-learn style interface for the tokenizer.
    
    Example:
        >>> tokenizer = BPETokenizer(vocab_size=5000)
        >>> tokenizer.fit(corpus)
        >>> encoded = tokenizer.encode(text)
        >>> decoded = tokenizer.decode(encoded)
    """
    
    def __init__(self, vocab_size: int = 5000, verbose: bool = True):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            verbose: Whether to print training progress
        """
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.vocab = None
        self.merge_rules = None
        self.is_trained = False
    
    def fit(self, corpus: str):
        """
        Train the tokenizer on a corpus.
        
        Args:
            corpus: DNA sequence string
        """
        self.vocab, self.merge_rules, _ = train_bpe(
            corpus, 
            self.vocab_size, 
            verbose=self.verbose
        )
        self.is_trained = True
        return self
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: DNA sequence string
            
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding. Call .fit() first.")
        
        return encode(text, self.vocab, self.merge_rules)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            DNA sequence string
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding. Call .fit() first.")
        
        return decode(token_ids, self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first.")
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """Get the actual vocabulary size."""
        if not self.is_trained:
            return 0
        return len(self.vocab)
    
    def __repr__(self):
        if self.is_trained:
            return f"BPETokenizer(vocab_size={len(self.vocab)}, trained=True)"
        else:
            return f"BPETokenizer(vocab_size={self.vocab_size}, trained=False)"