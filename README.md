# BPE Tokenizer Implementation

A complete implementation of Byte Pair Encoding (BPE) tokenization similar to that used in ChatGPT and other modern Large Language Models (LLMs). This project provides a deep understanding of how tokenization works in practice.

## What is BPE Tokenization?

Byte Pair Encoding (BPE) is a subword tokenization algorithm that:

1. **Starts with individual bytes** as the initial vocabulary
2. **Iteratively merges** the most frequent pair of consecutive tokens
3. **Builds a vocabulary** of subword units that efficiently represent text
4. **Enables handling** of out-of-vocabulary words through subword decomposition

This approach allows models to handle any text while maintaining a reasonable vocabulary size, making it ideal for LLMs that need to process diverse text from the internet.

## Key Features

- **GPT-style Implementation**: Follows the same BPE algorithm used in GPT models
- **Byte-level Encoding**: Handles any Unicode text robustly
- **Special Tokens**: Support for `<|endoftext|>`, `<|startoftext|>`, etc.
- **Pre-tokenization**: Uses regex patterns similar to GPT tokenizers
- **Save/Load**: Persistence for trained tokenizers
- **Analysis Tools**: Detailed vocabulary and performance analysis
- **Custom Training**: Train on your own datasets

## How It Works

### 1. Pre-tokenization

Text is first split into words using a regex pattern that handles:

- Contractions (`'s`, `'t`, `'re`, etc.)
- Letters and numbers
- Punctuation and whitespace

### 2. Byte Encoding

Each character is converted to bytes, then mapped to a special Unicode representation to avoid issues with invalid UTF-8 sequences.

### 3. BPE Training

The algorithm iteratively finds the most frequent pair of adjacent tokens and merges them:

```
Initial: ['h', 'e', 'l', 'l', 'o']
Step 1:  ['he', 'l', 'l', 'o']    # 'h' + 'e' was most frequent
Step 2:  ['he', 'll', 'o']        # 'l' + 'l' was most frequent
Step 3:  ['hell', 'o']            # 'he' + 'll' was most frequent
Final:   ['hello']                # 'hell' + 'o' was most frequent
```

### 4. Encoding/Decoding

- **Encoding**: Apply learned BPE merges to convert text to token IDs
- **Decoding**: Convert token IDs back to text using the vocabulary

## Installation and Usage

### Basic Example

```python
from bpe_tokenizer import BPETokenizer

# Create and train tokenizer
tokenizer = BPETokenizer()

# Sample training data
texts = [
    "Hello world! This is a test.",
    "Machine learning is fascinating.",
    "The quick brown fox jumps over the lazy dog.",
    # ... more training texts
]

# Train with target vocabulary size
tokenizer.train(texts, vocab_size=1000)

# Encode text to token IDs
text = "Hello, how are you?"
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# Save trained tokenizer
tokenizer.save("my_tokenizer.pkl")

# Load later
new_tokenizer = BPETokenizer()
new_tokenizer.load("my_tokenizer.pkl")
```

### Run the Demo

See the tokenizer in action with sample data:

```bash
python example_usage.py
```

This will show:

- Training process with progress updates
- Vocabulary analysis and learned subwords
- Encoding/decoding examples
- Comparison with simple word tokenization
- Step-by-step BPE merging process

### Train on Custom Data

Train a tokenizer on your own text data:

```bash
# Train on a single text file
python train_custom_tokenizer.py --input data.txt --vocab_size 10000 --output my_tokenizer.pkl

# Train on all text files in a directory
python train_custom_tokenizer.py --input ./text_data/ --vocab_size 50000 --output large_tokenizer.pkl

# Show detailed training progress
python train_custom_tokenizer.py --input data.txt --vocab_size 5000 --verbose --output tokenizer.pkl
```

Options:

- `--input`: Text file or directory containing text files
- `--vocab_size`: Target vocabulary size (default: 50,000)
- `--output`: Output file for trained tokenizer (default: tokenizer.pkl)
- `--max_files`: Maximum number of files to load from directory
- `--test_ratio`: Ratio of data to use for testing (default: 0.1)
- `--verbose`: Show detailed training progress

## Understanding the Implementation

### Core Algorithm

The heart of BPE is in the training loop:

```python
while len(vocab) < vocab_size:
    # Count all adjacent pairs across all words
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        word_pairs = self._get_pairs(list(word))
        for pair in word_pairs:
            pairs[pair] += freq

    # Find most frequent pair
    best_pair = max(pairs, key=pairs.get)

    # Merge this pair in all words
    # ... merge logic ...

    # Add merged token to vocabulary
    vocab.add(''.join(best_pair))
```

### Key Design Decisions

1. **Byte-level**: Works with bytes rather than characters, ensuring any text can be represented
2. **Deterministic**: Same input always produces same output
3. **Frequency-based**: Merges are based on actual frequency in training data
4. **Greedy**: Always merges the most frequent pair at each step

### Special Token Handling

The tokenizer includes common special tokens used in LLMs:

- `<|endoftext|>`: Marks end of documents/sequences
- `<|startoftext|>`: Marks start of documents/sequences
- `<|pad|>`: Padding token for batching
- `<|unk|>`: Unknown token for rare words

## Performance Analysis

The implementation provides detailed analysis of tokenizer performance:

### Vocabulary Statistics

- Distribution of token types (single chars, subwords, special tokens)
- Most frequent learned merges
- Vocabulary coverage analysis

### Compression Metrics

- Characters per token ratio
- Tokens per word ratio
- Overall compression ratio vs simple word tokenization

### Example Output

```
=== Vocabulary Analysis ===
Total vocabulary size: 1000
Number of BPE merges: 744

Token categories:
  Single characters: 256
  Subwords: 740
  Special tokens: 4

Compression ratio: 2.3x better than word tokenization
Average chars per token: 3.7
```

## Comparison with Production Tokenizers

This implementation follows the same core algorithm as production tokenizers like:

- **GPT-2/GPT-3 Tokenizer**: Same BPE algorithm and byte-level encoding
- **GPT-4 Tokenizer**: Similar approach with larger vocabulary
- **BERT WordPiece**: Different algorithm but similar subword concept

Key similarities:

- Byte-level encoding for robustness
- Pre-tokenization with regex patterns
- Iterative pair merging
- Subword vocabulary construction

## Educational Value

This implementation helps you understand:

1. **How modern LLMs handle text**: The actual tokenization process used by GPT and similar models
2. **Subword algorithms**: Why and how subword tokenization works
3. **Vocabulary construction**: How to build efficient vocabularies from data
4. **Text preprocessing**: Critical steps before model training
5. **Implementation details**: Real-world considerations like byte encoding and special tokens

## Advanced Usage

### Custom Special Tokens

```python
tokenizer = BPETokenizer()
# Add custom special tokens before training
tokenizer.special_tokens['<|code|>'] = None
tokenizer.special_tokens['<|data|>'] = None
```

### Analyzing Learned Patterns

```python
# Examine what the tokenizer learned
vocab = tokenizer.get_vocab()

# Find tokens containing specific patterns
programming_tokens = [token for token in vocab if 'def' in token or 'class' in token]
print(f"Programming-related tokens: {programming_tokens}")

# Analyze merge operations
for i, (token1, token2) in enumerate(tokenizer.bpe_merges[-10:]):
    print(f"Merge {i}: '{token1}' + '{token2}' -> '{token1 + token2}'")
```

### Performance Optimization

For large datasets:

- Use `max_files` parameter to limit training data
- Increase `chunk_size` in training script for better memory usage
- Train with smaller vocabulary first to test, then scale up

## Files Description

- `bpe_tokenizer.py`: Core BPE tokenizer implementation
- `example_usage.py`: Comprehensive demo with sample data
- `train_custom_tokenizer.py`: Command-line training script
- `README.md`: This documentation

## Further Reading

- [Original BPE Paper](https://arxiv.org/abs/1508.07909): Sennrich et al., 2015
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): Radford et al., 2019
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/): Production tokenizer library

## Contributing

This is an educational implementation. Feel free to:

- Experiment with different pre-tokenization patterns
- Add new special tokens for specific domains
- Optimize training performance
- Compare with other tokenization algorithms

The goal is learning and understanding, so explore and modify as needed!
