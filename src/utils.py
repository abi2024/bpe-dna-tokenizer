"""
Utility Functions

Helper functions for analysis, visualization, and I/O operations.
"""

import pickle
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_compression_ratio(original_corpus: str, 
                               compressed_tokens: List[str]) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_corpus: Original DNA string
        compressed_tokens: List of tokens after BPE
        
    Returns:
        Compression ratio (original_size / compressed_size)
        
    Example:
        >>> ratio = calculate_compression_ratio(corpus, tokens)
        >>> print(f"Compression: {ratio:.2f}x")
    """
    original_size = len(original_corpus)
    compressed_size = len(compressed_tokens)
    
    if compressed_size == 0:
        return 0.0
    
    return original_size / compressed_size


def save_tokenizer(vocab: Dict[str, int], 
                   merge_rules: List[Tuple[str, str]], 
                   filename: str = "models/ecoli_bpe_tokenizer.pkl",
                   save_text: bool = True):
    """
    Save trained tokenizer to file.
    
    Args:
        vocab: Vocabulary dictionary
        merge_rules: List of merge rules
        filename: Output filename (.pkl)
        save_text: Also save human-readable text version
        
    Example:
        >>> save_tokenizer(vocab, merge_rules, "my_tokenizer.pkl")
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tokenizer = {
        'vocab': vocab,
        'merge_rules': merge_rules,
        'vocab_size': len(vocab),
        'num_merges': len(merge_rules),
    }
    
    # Save binary pickle
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"✓ Tokenizer saved to {filename}")
    print(f"  - Vocabulary size: {len(vocab):,}")
    print(f"  - Number of merge rules: {len(merge_rules):,}")
    
    # Save text version
    if save_text:
        text_filename = filename.replace('.pkl', '.txt')
        
        with open(text_filename, 'w') as f:
            f.write("BPE TOKENIZER\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Vocabulary Size: {len(vocab)}\n")
            f.write(f"Number of Merges: {len(merge_rules)}\n\n")
            
            f.write("VOCABULARY (first 50 tokens):\n")
            f.write("-"*60 + "\n")
            for i, (token, idx) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:50]):
                f.write(f"{idx:>5}: {token}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"✓ Text version saved to {text_filename}")


def load_tokenizer(filename: str = "models/ecoli_bpe_tokenizer.pkl") -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Load tokenizer from file.
    
    Args:
        filename: Path to tokenizer file (.pkl)
        
    Returns:
        Tuple of (vocab, merge_rules)
        
    Example:
        >>> vocab, rules = load_tokenizer("my_tokenizer.pkl")
    """
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"✓ Tokenizer loaded from {filename}")
    print(f"  - Vocabulary size: {tokenizer['vocab_size']:,}")
    print(f"  - Number of merge rules: {tokenizer['num_merges']:,}")
    
    return tokenizer['vocab'], tokenizer['merge_rules']


def analyze_vocabulary(vocab: Dict[str, int], 
                       merge_rules: List[Tuple[str, str]],
                       output_file: Optional[str] = None) -> Dict:
    """
    Analyze learned vocabulary patterns.
    
    Args:
        vocab: Vocabulary dictionary
        merge_rules: List of merge rules
        output_file: Optional file to save analysis
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Token length distribution
    token_lengths = [len(token) for token in vocab.keys()]
    analysis['token_lengths'] = {
        'min': min(token_lengths),
        'max': max(token_lengths),
        'mean': sum(token_lengths) / len(token_lengths),
        'distribution': dict(Counter(token_lengths)),
    }
    
    # Character frequency in vocab
    char_counts = Counter()
    for token in vocab.keys():
        char_counts.update(token)
    analysis['char_frequency'] = dict(char_counts)
    
    # Longest tokens
    sorted_by_length = sorted(vocab.keys(), key=len, reverse=True)
    analysis['longest_tokens'] = sorted_by_length[:20]
    
    # First merges (most common patterns)
    analysis['first_merges'] = merge_rules[:20]
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Analysis saved to {output_file}")
    
    return analysis


def find_biological_patterns(vocab: Dict[str, int]) -> Dict[str, List[str]]:
    """
    Search for biologically meaningful patterns in vocabulary.
    
    Args:
        vocab: Vocabulary dictionary
        
    Returns:
        Dictionary of pattern categories and found patterns
        
    Example:
        >>> patterns = find_biological_patterns(vocab)
        >>> print(f"Found start codons: {patterns['Start Codons']}")
    """
    patterns = {
        'Start Codons': ['ATG'],
        'Stop Codons': ['TAA', 'TAG', 'TGA'],
        'TATA Box': ['TATA', 'TATAA', 'TATAAA', 'TATATAT'],
        'Shine-Dalgarno': ['AGGAGG', 'AGGA', 'GGAGG'],
        'Poly-A': ['AAA', 'AAAA', 'AAAAA', 'AAAAAA'],
        'Poly-T': ['TTT', 'TTTT', 'TTTTT', 'TTTTTT'],
        'CpG': ['CG', 'CGCG', 'GCGC'],
        'Homopolymers': ['GGG', 'GGGG', 'CCC', 'CCCC'],
    }
    
    found_patterns = {}
    
    for category, pattern_list in patterns.items():
        found = [p for p in pattern_list if p in vocab]
        if found:
            found_patterns[category] = found
    
    return found_patterns


def plot_results(vocab: Dict[str, int],
                compressed_tokens: List[str],
                corpus: str,
                output_dir: str = "results/figures"):
    """
    Create visualizations of BPE training results.
    
    Args:
        vocab: Vocabulary dictionary
        compressed_tokens: Compressed token list
        corpus: Original corpus
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 4)
    
    # 1. Token length distribution
    token_lengths = [len(token) for token in vocab.keys()]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(token_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Length (bases)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths')
    plt.axvline(sum(token_lengths)/len(token_lengths), 
                color='red', linestyle='--', 
                label=f'Mean: {sum(token_lengths)/len(token_lengths):.2f}')
    plt.legend()
    
    # 2. Compression comparison
    plt.subplot(1, 3, 2)
    sizes = [len(corpus), len(compressed_tokens)]
    labels = ['Original\n(bytes)', 'Compressed\n(tokens)']
    colors = ['#ff6b6b', '#4ecdc4']
    bars = plt.bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Size')
    plt.title('Compression Comparison')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:,}',
                ha='center', va='bottom')
    
    # 3. Compression ratio
    plt.subplot(1, 3, 3)
    ratio = len(corpus) / len(compressed_tokens)
    plt.bar(['Compression\nRatio'], [ratio], color='#95e1d3', alpha=0.7, edgecolor='black')
    plt.ylabel('Ratio')
    plt.title('Compression Ratio')
    plt.axhline(y=3.2, color='red', linestyle='--', label='Target: 3.2')
    plt.text(0, ratio, f'{ratio:.3f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bpe_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.show()


def save_statistics(vocab: Dict[str, int],
                   compressed_tokens: List[str],
                   corpus: str,
                   training_time: float,
                   output_file: str = "results/statistics/compression_stats.json"):
    """
    Save compression statistics to JSON file.
    
    Args:
        vocab: Vocabulary dictionary
        compressed_tokens: Compressed token list
        corpus: Original corpus
        training_time: Training time in seconds
        output_file: Output JSON file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    stats = {
        'vocab_size': len(vocab),
        'compression_ratio': len(corpus) / len(compressed_tokens),
        'training_time_seconds': training_time,
        'original_size_bytes': len(corpus),
        'compressed_size_tokens': len(compressed_tokens),
        'average_token_length': len(corpus) / len(compressed_tokens),
        'max_token_length': max(len(token) for token in vocab.keys()),
        'min_token_length': min(len(token) for token in vocab.keys()),
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to {output_file}")
    
    return stats


def print_summary(vocab: Dict[str, int],
                 compressed_tokens: List[str],
                 corpus: str,
                 training_time: float):
    """
    Print a formatted summary of results.
    """
    compression_ratio = len(corpus) / len(compressed_tokens)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Vocabulary size:      {len(vocab):,} tokens")
    print(f"Original size:        {len(corpus):,} bytes")
    print(f"Compressed size:      {len(compressed_tokens):,} tokens")
    print(f"Compression ratio:    {compression_ratio:.3f}x")
    print(f"Average token length: {len(corpus) / len(compressed_tokens):.3f} bases")
    print(f"Training time:        {training_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("REQUIREMENTS CHECK")
    print("="*60)
    vocab_pass = "✓ PASS" if len(vocab) >= 5000 else "✗ FAIL"
    compression_pass = "✓ PASS" if compression_ratio >= 3.2 else "✗ FAIL"
    
    print(f"{vocab_pass} Vocabulary size ≥ 5000: {len(vocab):,}")
    print(f"{compression_pass} Compression ratio ≥ 3.2: {compression_ratio:.3f}")
    
    if len(vocab) >= 5000 and compression_ratio >= 3.2:
        print("\n🎉 ALL REQUIREMENTS MET! 🎉")
    else:
        print("\n⚠️  Some requirements not met.")
    
    print("="*60)