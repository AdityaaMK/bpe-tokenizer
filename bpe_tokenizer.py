"""
BPE (Byte Pair Encoding) Tokenizer Implementation

This implementation follows the BPE algorithm used in GPT and other modern LLMs.
It starts with individual bytes and iteratively merges the most frequent pairs.

Very similar to actual GPT tokenizer implementation:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import pickle
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer similar to those used in ChatGPT and other LLMs.

    The tokenizer works by:
    1. Starting with individual bytes as the initial vocabulary
    2. Iteratively merging the most frequent pair of consecutive tokens
    3. Building a vocabulary of subword units for efficient encoding
    """

    def __init__(self):
        self.encoder: Dict[str, int] = {}  # token -> id mapping
        self.decoder: Dict[int, str] = {}  # id -> token mapping
        # ordered list of merge operations
        self.bpe_merges: List[Tuple[str, str]] = []
        # merge pair -> rank mapping
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.vocab_size = 0

        # Special tokens commonly used in LLMs
        self.special_tokens = {
            '<|endoftext|>': None,  # End of document/sequence
            '<|startoftext|>': None,  # Start of document/sequence
            '<|pad|>': None,  # Padding token
            '<|unk|>': None,  # Unknown token
        }

        # Pattern for pre-tokenization (Python-compatible regex)
        # This pattern handles: contractions, letters, numbers, punctuation, whitespace
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")

    def _get_byte_encoder(self) -> Dict[int, str]:
        """
        Create a mapping from bytes to unicode strings.
        This avoids issues with bytes that are not valid UTF-8.
        """
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"),
                                                            ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_byte_decoder(self) -> Dict[str, int]:
        """Create reverse mapping from unicode strings to bytes."""
        byte_encoder = self._get_byte_encoder()
        return {v: k for k, v in byte_encoder.items()}

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text into words using regex pattern.
        This is similar to how GPT tokenizers work.
        """
        return re.findall(self.pat, text)

    def train(self, texts: List[str], vocab_size: int = 50000, verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            verbose: Whether to print training progress
        """
        if verbose:
            print(
                f"Training BPE tokenizer with target vocab size: {vocab_size}")

        # Initialize byte encoder/decoder
        self.byte_encoder = self._get_byte_encoder()
        self.byte_decoder = self._get_byte_decoder()

        # Pre-tokenize all texts and convert to bytes
        if verbose:
            print("Pre-tokenizing texts...")

        all_words = []
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                # Convert to bytes and then to our byte encoding
                word_bytes = word.encode('utf-8')
                word_tokens = [self.byte_encoder[b] for b in word_bytes]
                all_words.append(word_tokens)

        # Count word frequencies
        word_freqs = Counter(tuple(word) for word in all_words)

        # Initialize vocabulary with all byte tokens
        vocab = set()
        for word in word_freqs:
            vocab.update(word)

        # Add special tokens to vocabulary
        for special_token in self.special_tokens:
            vocab.add(special_token)

        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")

        # Perform BPE merges
        merges = []
        while len(vocab) < vocab_size:
            # Count all pairs across all words
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_pairs = self._get_pairs(list(word))
                for pair in word_pairs:
                    pairs[pair] += freq

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)

            # Merge the best pair in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_word(list(word), best_pair)
                new_word_freqs[tuple(new_word)] = freq

            word_freqs = new_word_freqs
            vocab.add(''.join(best_pair))

            if verbose and len(merges) % 1000 == 0:
                print(
                    f"Learned {len(merges)} merges, vocab size: {len(vocab)}")

        # Store the merges and create mappings
        self.bpe_merges = merges
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # Create encoder/decoder mappings
        vocab_list = sorted(vocab)
        self.encoder = {token: i for i, token in enumerate(vocab_list)}
        self.decoder = {i: token for token, i in self.encoder.items()}
        self.vocab_size = len(vocab_list)

        # Update special token IDs
        for special_token in self.special_tokens:
            if special_token in self.encoder:
                self.special_tokens[special_token] = self.encoder[special_token]

        if verbose:
            print(f"Training complete! Final vocab size: {self.vocab_size}")
            print(f"Learned {len(self.bpe_merges)} BPE merges")

    def _merge_word(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge all instances of a pair in a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def _bpe_encode(self, word: str) -> List[str]:
        """Apply BPE encoding to a single word."""
        if not word:
            return []

        # Convert word to byte representation
        word_bytes = word.encode('utf-8')
        word_tokens = [self.byte_encoder[b] for b in word_bytes]

        if len(word_tokens) == 1:
            return word_tokens

        pairs = self._get_pairs(word_tokens)

        if not pairs:
            return word_tokens

        while True:
            # Find the highest priority pair to merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))

            if bigram not in self.bpe_ranks:
                break

            # Merge the pair
            word_tokens = self._merge_word(word_tokens, bigram)

            if len(word_tokens) == 1:
                break

            pairs = self._get_pairs(word_tokens)

        return word_tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        if not hasattr(self, 'encoder') or not self.encoder:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Pre-tokenize text
        words = self._pre_tokenize(text)

        # Encode each word with BPE
        token_ids = []
        for word in words:
            bpe_tokens = self._bpe_encode(word)
            for token in bpe_tokens:
                if token in self.encoder:
                    token_ids.append(self.encoder[token])
                else:
                    # Use unknown token if available
                    if '<|unk|>' in self.special_tokens and self.special_tokens['<|unk|>'] is not None:
                        token_ids.append(self.special_tokens['<|unk|>'])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not hasattr(self, 'decoder') or not self.decoder:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Convert token IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                tokens.append(self.decoder[token_id])

        # Join tokens and decode bytes
        text = ''.join(tokens)

        # Convert back from byte encoding to text
        try:
            byte_sequence = [self.byte_decoder[c]
                             for c in text if c in self.byte_decoder]
            return bytes(byte_sequence).decode('utf-8', errors='replace')
        except:
            return text

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        return self.encoder.copy()

    def save(self, filepath: str) -> None:
        """Save the trained tokenizer to disk."""
        tokenizer_data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'bpe_merges': self.bpe_merges,
            'bpe_ranks': self.bpe_ranks,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'byte_encoder': self.byte_encoder,
            'byte_decoder': self.byte_decoder,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)

        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load a trained tokenizer from disk."""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)

        self.encoder = tokenizer_data['encoder']
        self.decoder = tokenizer_data['decoder']
        self.bpe_merges = tokenizer_data['bpe_merges']
        self.bpe_ranks = tokenizer_data['bpe_ranks']
        self.vocab_size = tokenizer_data['vocab_size']
        self.special_tokens = tokenizer_data['special_tokens']
        self.byte_encoder = tokenizer_data['byte_encoder']
        self.byte_decoder = tokenizer_data['byte_decoder']

        print(f"Tokenizer loaded from {filepath}")
