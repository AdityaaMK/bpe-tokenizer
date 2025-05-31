"""
Example usage of the BPE Tokenizer

This script demonstrates how to:
1. Train a BPE tokenizer on sample text data
2. Encode and decode text
3. Analyze the learned vocabulary
4. Save and load trained tokenizers
"""

from bpe_tokenizer import BPETokenizer
import time


def get_sample_training_data():
    """Get sample training data for demonstration."""

    # Sample texts covering various domains and styles
    sample_texts = [
        # Literature/narrative
        """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. 
        In the heart of the city, where skyscrapers touched the clouds and the streets buzzed with life, 
        there lived a programmer named Alice who dreamed of building artificial intelligence.""",

        # Technical content
        """Machine learning models require large amounts of training data to perform well. The transformer 
        architecture has revolutionized natural language processing. Attention mechanisms allow models to 
        focus on relevant parts of the input sequence. Tokenization is a crucial preprocessing step.""",

        # Conversational
        """Hey! How are you doing today? I'm pretty good, thanks for asking. Did you see the news about 
        the new AI model? It's supposed to be really impressive. I can't wait to try it out myself. 
        What do you think about it?""",

        # Code-like content
        """def tokenize_text(text): return text.split() 
        class BPETokenizer: def __init__(self): self.vocab = {} 
        import torch import numpy as np from transformers import GPT2Tokenizer""",

        # Mixed content with punctuation and numbers
        """The year 2024 marked a significant milestone in AI development. GPT-4 achieved unprecedented 
        performance on various benchmarks, scoring 95.3% on the HellaSwag dataset and 92.1% on MMLU. 
        The model contains approximately 1.76 trillion parameters, making it one of the largest ever created.""",

        # Repeat some patterns to establish common subwords
        """Hello world! Hello everyone! Hello there! Programming is fun. Programming is challenging. 
        Programming requires practice. The algorithm processes data efficiently. The algorithm learns patterns. 
        The algorithm makes predictions."""
    ]

    return sample_texts


def demonstrate_tokenizer():
    """Demonstrate the BPE tokenizer functionality."""

    print("=== BPE Tokenizer Demonstration ===\n")

    # 1. Create and train tokenizer
    print("1. Training the tokenizer...")
    print("-" * 40)

    tokenizer = BPETokenizer()
    training_texts = get_sample_training_data()

    print(f"Training on {len(training_texts)} text samples...")
    start_time = time.time()

    # Train with a smaller vocabulary for demonstration
    tokenizer.train(training_texts, vocab_size=1000, verbose=True)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds\n")

    # 2. Analyze the vocabulary
    print("2. Analyzing the learned vocabulary...")
    print("-" * 40)

    vocab = tokenizer.get_vocab()
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of BPE merges learned: {len(tokenizer.bpe_merges)}")

    # Show some example tokens
    example_tokens = list(vocab.keys())[:20]
    print(f"\nFirst 20 tokens in vocabulary:")
    for i, token in enumerate(example_tokens):
        print(f"  {i:2d}: '{token}' -> {vocab[token]}")

    # Show some merged tokens (subwords)
    merged_tokens = [token for token in vocab.keys() if len(
        token) > 1 and not token.startswith('<|')]
    print(f"\nSome learned subword tokens:")
    for i, token in enumerate(merged_tokens[:10]):
        print(f"  '{token}' -> {vocab[token]}")

    print()

    # 3. Test encoding and decoding
    print("3. Testing encoding and decoding...")
    print("-" * 40)

    test_sentences = [
        "Hello, world! This is a test.",
        "The algorithm processes data efficiently.",
        "Machine learning models are powerful tools.",
        "Programming is both fun and challenging!",
    ]

    for sentence in test_sentences:
        print(f"\nOriginal: '{sentence}'")

        # Encode
        encoded = tokenizer.encode(sentence)
        print(f"Encoded:  {encoded}")
        print(f"Length:   {len(encoded)} tokens")

        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded:  '{decoded}'")

        # Check if encoding/decoding is lossless
        matches = sentence == decoded
        print(f"Lossless: {matches}")

        if not matches:
            print(f"  Difference detected!")

    # 4. Compare with simple whitespace tokenization
    print("\n4. Comparison with whitespace tokenization...")
    print("-" * 40)

    comparison_text = "The transformer architecture revolutionized natural language processing."

    # BPE tokenization
    bpe_tokens = tokenizer.encode(comparison_text)
    bpe_token_strs = [tokenizer.decoder[tid] for tid in bpe_tokens]

    # Simple whitespace tokenization
    simple_tokens = comparison_text.split()

    print(f"Text: '{comparison_text}'")
    print(f"\nBPE tokens ({len(bpe_tokens)}): {bpe_token_strs}")
    print(f"Simple tokens ({len(simple_tokens)}): {simple_tokens}")

    # 5. Test special tokens
    print("\n5. Special tokens...")
    print("-" * 40)

    print("Special tokens in vocabulary:")
    for special_token, token_id in tokenizer.special_tokens.items():
        if token_id is not None:
            print(f"  '{special_token}' -> {token_id}")

    # 6. Save and load tokenizer
    print("\n6. Saving and loading tokenizer...")
    print("-" * 40)

    # Save tokenizer
    save_path = "trained_tokenizer.pkl"
    tokenizer.save(save_path)

    # Load tokenizer
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(save_path)

    # Test that loaded tokenizer works the same
    test_text = "Testing the loaded tokenizer."
    original_encoded = tokenizer.encode(test_text)
    loaded_encoded = new_tokenizer.encode(test_text)

    print(f"Original tokenizer encoding: {original_encoded}")
    print(f"Loaded tokenizer encoding:   {loaded_encoded}")
    print(f"Encodings match: {original_encoded == loaded_encoded}")

    print("\n=== Demonstration Complete ===")


def analyze_bpe_process():
    """Analyze how BPE merging works step by step."""

    print("\n=== BPE Merging Process Analysis ===\n")

    # Create a simple example to show BPE merging
    simple_texts = [
        "low lower lowest",
        "new newer newest",
        "the the the",
        "and and and",
    ] * 20  # Repeat to increase frequency

    tokenizer = BPETokenizer()
    tokenizer.train(simple_texts, vocab_size=300, verbose=False)

    print("Example of learned BPE merges:")
    print("-" * 40)

    # Show first 10 merges
    for i, (token1, token2) in enumerate(tokenizer.bpe_merges[:10]):
        merged = token1 + token2
        print(f"Merge {i+1:2d}: '{token1}' + '{token2}' -> '{merged}'")

    # Show how a word gets tokenized step by step
    print(f"\nStep-by-step tokenization of 'lowest':")
    print("-" * 40)

    word = "lowest"
    print(f"Original word: '{word}'")

    # Start with byte encoding
    word_bytes = word.encode('utf-8')
    word_tokens = [tokenizer.byte_encoder[b] for b in word_bytes]
    print(f"Byte tokens: {word_tokens}")

    # Apply BPE step by step (simplified view)
    current_tokens = word_tokens[:]
    step = 1

    while len(current_tokens) > 1:
        # Find pairs and their ranks
        pairs = []
        for i in range(len(current_tokens) - 1):
            pair = (current_tokens[i], current_tokens[i + 1])
            if pair in tokenizer.bpe_ranks:
                pairs.append((pair, tokenizer.bpe_ranks[pair]))

        if not pairs:
            break

        # Find best pair (lowest rank = earliest learned)
        best_pair, rank = min(pairs, key=lambda x: x[1])

        # Apply merge
        new_tokens = []
        i = 0
        while i < len(current_tokens):
            if i < len(current_tokens) - 1 and (current_tokens[i], current_tokens[i + 1]) == best_pair:
                new_tokens.append(current_tokens[i] + current_tokens[i + 1])
                i += 2
            else:
                new_tokens.append(current_tokens[i])
                i += 1

        print(
            f"Step {step}: {current_tokens} -> {new_tokens} (merged '{best_pair[0]}' + '{best_pair[1]}')")
        current_tokens = new_tokens
        step += 1

        if step > 10:  # Prevent infinite loops
            break

    print(f"Final tokens: {current_tokens}")


if __name__ == "__main__":
    demonstrate_tokenizer()
    analyze_bpe_process()
