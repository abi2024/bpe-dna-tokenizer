"""
Data Loading Utilities

Functions for loading and preprocessing DNA sequence data.
"""

import gzip
import os
import requests
from typing import Dict
from collections import Counter


def download_ecoli_genome(output_dir: str = "data/raw", 
                          force_download: bool = False) -> str:
    """
    Download E. coli genome from NCBI.
    
    Args:
        output_dir: Directory to save the file
        force_download: Re-download even if file exists
        
    Returns:
        Path to downloaded file
        
    Example:
        >>> filepath = download_ecoli_genome()
        >>> print(f"Downloaded to: {filepath}")
    """
    ECOLI_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, "ecoli_genome.fna.gz")
    
    # Check if file already exists
    if os.path.exists(filename) and not force_download:
        print(f"✓ {filename} already exists (use force_download=True to re-download)")
        return filename
    
    print("Downloading E. coli genome from NCBI...")
    print(f"URL: {ECOLI_URL}")
    
    try:
        response = requests.get(ECOLI_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress indicator
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
                print()  # New line after progress
            else:
                f.write(response.content)
        
        print(f"✓ Downloaded successfully: {filename}")
        print(f"✓ File size: {os.path.getsize(filename):,} bytes")
        
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise
    
    return filename


def load_fasta(filename: str, validate: bool = True) -> str:
    """
    Load FASTA file and return DNA sequence as a single string.
    
    Supports both compressed (.gz) and uncompressed files.
    
    Args:
        filename: Path to FASTA file (.fna, .fasta, .fa, or .gz)
        validate: Whether to validate the sequence contains only valid bases
        
    Returns:
        DNA sequence as uppercase string
        
    Example:
        >>> corpus = load_fasta("data/raw/ecoli_genome.fna.gz")
        >>> print(f"Loaded {len(corpus):,} base pairs")
    """
    sequences = []
    
    # Determine if file is compressed
    is_gzipped = filename.endswith('.gz')
    
    try:
        # Open file (gzip or regular)
        if is_gzipped:
            file_handle = gzip.open(filename, 'rt')
        else:
            file_handle = open(filename, 'r')
        
        with file_handle as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip header lines (start with '>')
                if line.startswith('>'):
                    continue
                
                # This is a sequence line
                sequences.append(line.upper())
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {e}")
    
    # Concatenate all sequences
    corpus = ''.join(sequences)
    
    # Validate sequence
    if validate:
        valid_bases = set('ACGTN')
        invalid_chars = set(corpus) - valid_bases
        
        if invalid_chars:
            print(f"Warning: Found invalid characters in sequence: {invalid_chars}")
            print(f"Valid DNA bases are: {valid_bases}")
            
            # Option to clean the sequence
            corpus = ''.join(c for c in corpus if c in valid_bases)
            print(f"Cleaned sequence: {len(corpus):,} bases")
    
    return corpus


def get_sequence_stats(sequence: str) -> Dict:
    """
    Calculate statistics about a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with sequence statistics
        
    Example:
        >>> stats = get_sequence_stats(corpus)
        >>> print(f"GC content: {stats['gc_content']:.2%}")
    """
    # Character counts
    char_counts = Counter(sequence)
    total_bases = len(sequence)
    
    # Calculate GC content
    gc_count = char_counts.get('G', 0) + char_counts.get('C', 0)
    gc_content = gc_count / total_bases if total_bases > 0 else 0
    
    # Calculate AT content
    at_count = char_counts.get('A', 0) + char_counts.get('T', 0)
    at_content = at_count / total_bases if total_bases > 0 else 0
    
    stats = {
        'length': total_bases,
        'base_counts': dict(char_counts),
        'gc_content': gc_content,
        'at_content': at_content,
        'n_count': char_counts.get('N', 0),
        'unique_chars': len(char_counts),
    }
    
    return stats


def print_sequence_stats(sequence: str):
    """
    Print formatted statistics about a DNA sequence.
    
    Args:
        sequence: DNA sequence string
    """
    stats = get_sequence_stats(sequence)
    
    print("="*60)
    print("SEQUENCE STATISTICS")
    print("="*60)
    print(f"Total length:        {stats['length']:,} base pairs")
    print(f"Unique characters:   {stats['unique_chars']}")
    print(f"\nBase composition:")
    
    for base in sorted(stats['base_counts'].keys()):
        count = stats['base_counts'][base]
        percentage = (count / stats['length']) * 100
        print(f"  {base}: {count:>12,} ({percentage:>5.2f}%)")
    
    print(f"\nGC content:          {stats['gc_content']:.2%}")
    print(f"AT content:          {stats['at_content']:.2%}")
    
    if stats['n_count'] > 0:
        print(f"\nAmbiguous bases (N): {stats['n_count']:,}")
    
    print("="*60)


def sample_sequence(sequence: str, sample_size: int, seed: int = 42) -> str:
    """
    Sample a random contiguous subsequence.
    
    Useful for testing on smaller datasets.
    
    Args:
        sequence: Full DNA sequence
        sample_size: Number of bases to sample
        seed: Random seed for reproducibility
        
    Returns:
        Sampled subsequence
        
    Example:
        >>> small_corpus = sample_sequence(corpus, sample_size=100000)
    """
    import random
    
    if sample_size >= len(sequence):
        return sequence
    
    random.seed(seed)
    start_pos = random.randint(0, len(sequence) - sample_size)
    end_pos = start_pos + sample_size
    
    return sequence[start_pos:end_pos]


def validate_fasta_file(filename: str) -> bool:
    """
    Validate that a file is a properly formatted FASTA file.
    
    Args:
        filename: Path to FASTA file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        is_gzipped = filename.endswith('.gz')
        
        if is_gzipped:
            file_handle = gzip.open(filename, 'rt')
        else:
            file_handle = open(filename, 'r')
        
        has_header = False
        has_sequence = False
        
        with file_handle as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('>'):
                    has_header = True
                elif line and not line.startswith('>'):
                    has_sequence = True
                
                if has_header and has_sequence:
                    return True
        
        return has_header and has_sequence
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False