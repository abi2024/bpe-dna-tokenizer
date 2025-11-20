import pytest
from src.bpe_tokenizer import get_pair_counts, merge_pair, train_bpe

def test_get_pair_counts():
    """Test pair counting functionality"""
    tokens = ['A', 'T', 'A', 'T']
    counts = get_pair_counts(tokens)
    assert counts[('A', 'T')] == 2
    assert counts[('T', 'A')] == 1

def test_merge_pair():
    """Test pair merging functionality"""
    tokens = ['A', 'T', 'C', 'A', 'T']
    result = merge_pair(tokens, ('A', 'T'), 'AT')
    assert result == ['AT', 'C', 'AT']

def test_lossless_encoding():
    """Test that encode->decode is lossless"""
    from src.bpe_tokenizer import encode, decode, initialize_vocab
    
    text = "ATCGATCGATCG"
    vocab = initialize_vocab(text)
    merge_rules = []  # Simple test without training
    
    encoded = encode(text, vocab, merge_rules)
    decoded = decode(encoded, vocab)
    
    assert text == decoded

def test_vocab_size_increases():
    """Test that vocabulary grows with training"""
    corpus = "ATCGATCGATCGATCG" * 100
    initial_vocab = initialize_vocab(corpus)
    
    vocab, rules, tokens = train_bpe(corpus, target_vocab_size=10)
    
    assert len(vocab) == 10
    assert len(rules) == 10 - len(initial_vocab)