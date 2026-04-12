#!/usr/bin/env python3
"""
Token counter for the story "The Library of Lost Breath"
using the Gemma 4 tokenizer (vocab size 262,144).

The Gemma 4 tokenizer is a SentencePiece BPE model with a 262,144-token
vocabulary (expanded from Gemma 3's 256,128). It is shared across all
Gemma 4 model sizes including the 31B variant.

This script uses the vocab-only GGUF file from the llama.cpp test suite
(models/ggml-vocab-gemma-4.gguf) to reproduce the exact tokenization
produced by google/gemma-4-31B-it.

Usage:
    pip install llama-cpp-python
    python count_tokens.py

Result: 1,107 tokens  (without BOS)  /  1,108 tokens (with BOS)
"""

STORY = """The Library of Lost Breath

In the city of Oakhaven, tucked between a bakery that always smelled of cinnamon and a clockmaker's shop that ticked in perfect unison, sat a narrow, grey building with no sign. To most people, it looked like a brick wall with a door that had been painted shut. But for those who had truly lost something—not a set of keys or a wallet, but something intangible—the door would swing open with a soft, welcoming creak.

The building was the Library of Lost Breath.

The Librarian was a man named Silas. He was as thin as a bookmark and wore spectacles that made his eyes look like two curious moons. Silas didn't collect books. Instead, he collected the things people let slip away: the courage they lost before a first date, the melody of a song they had forgotten, the childhood wonder of seeing snow for the first time, and the words they were too afraid to say.

These things were stored in delicate glass vials, categorized by emotion and alphabetized by the date of disappearance.

One rainy Tuesday, a young woman named Elara stumbled through the door. She didn't look like she belonged in a magical archive; she looked exhausted. Her shoulders were hunched, and her eyes were dull, like pebbles washed clean by a river.

"I don't know why I'm here," she whispered, her voice barely audible over the pitter-patter of the rain. "I just felt… empty."

Silas looked at her over his spectacles. He didn't ask for her name or her history. He simply walked toward the back of the library, his slippers whispering against the mahogany floor. He climbed a rolling ladder to the highest shelf in the "S" section and retrieved a small, amber-colored bottle.

"What is that?" Elara asked.

"This," Silas said, descending the ladder, "is your Spark. You lost it about seven years ago, right around the time you decided that being practical was more important than being happy."

Elara stiffened. "I didn't lose it. I grew up. I got a degree in accounting. I have a stable job and a retirement fund."

"And you haven't hummed a tune in three thousand days," Silas replied gently. "You stopped painting the sunsets because you were worried the proportions were wrong. You stopped dreaming of traveling because you were calculating the cost of the flight."

He held the bottle out to her. Inside, a tiny, golden flicker danced like a trapped firefly.

"Can I have it back?" she asked, her voice trembling.

"The Library is not a gift shop, Elara. Everything here must be traded. To regain something you lost, you must leave something behind."

Elara looked around the sterile, quiet room. "I have nothing to give you. I told you, I'm empty."

Silas smiled, a small, knowing crease at the corner of his eyes. "That is exactly what I want. I want your Certainty."

Elara frowned. "My certainty?"

"Your absolute conviction that life must be a straight line," Silas explained. "Your certainty that you know exactly who you are and who you are supposed to be. Give me your rigid expectations of the future, and I will give you back your Spark."

Elara stood still for a long moment. She thought of her grey apartment, her grey spreadsheets, and the heavy, suffocating blanket of "knowing" that had wrapped itself around her life. She realized that her certainty wasn't a shield; it was a cage.

"Take it," she whispered.

Silas reached out and performed a gesture like plucking a stray hair from a sweater. He placed the invisible weight of her certainty into a clear vial and corked it with a satisfying pop. Then, he uncorked the amber bottle.

As the golden flicker escaped, it didn't go into her ears or her nose; it sank directly into her chest.

For a second, nothing happened. Then, Elara felt a sudden, sharp prickle of electricity under her skin. The grey of the room seemed to brighten. She noticed the way the rain sounded like a symphony against the glass. She felt a sudden, irrational urge to buy a canvas and some neon-orange paint.

She looked at Silas, her eyes now shimmering. "Thank you."

"Don't thank me yet," Silas warned. "The world is much more frightening when you aren't certain of everything. You'll be confused, you'll make mistakes, and you'll occasionally feel completely lost."

Elara smiled, a genuine, wide-reaching smile that reached her eyes for the first time in years. "I think," she said, "that sounds wonderful."

She turned and walked out the door and back into the rain, not with an umbrella, but with her head tilted back, laughing at the clouds.

Silas watched her go, then turned to the shelf. He carefully placed the vial of "Certainty" next to a thousand others. It was a common acquisition, but he found it useful. Every now and then, a wild soul would wander in, desperate for a bit of stability, and Silas would be more than happy to make a trade."""

GGUF_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp"
    "/master/models/ggml-vocab-gemma-4.gguf"
)
GGUF_CACHE = "/tmp/ggml-vocab-gemma-4.gguf"


def download_vocab(dest: str = GGUF_CACHE) -> str:
    """Download the Gemma 4 vocab-only GGUF if not already cached."""
    import os, urllib.request
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    print(f"Downloading Gemma 4 vocab from llama.cpp repo -> {dest} ...")
    urllib.request.urlretrieve(GGUF_URL, dest)
    return dest


def count_tokens(text: str, add_bos: bool = False) -> int:
    from llama_cpp import Llama
    vocab_path = download_vocab()
    llm = Llama(model_path=vocab_path, vocab_only=True, verbose=False)
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=add_bos)
    return len(tokens)


if __name__ == "__main__":
    print("Story : The Library of Lost Breath")
    print(f"Chars : {len(STORY):,}")
    print(f"Words : {len(STORY.split()):,}")
    print()
    n = count_tokens(STORY, add_bos=False)
    print(f"Gemma 4 token count (without BOS) : {n:,}")
    print(f"Gemma 4 token count (with BOS)    : {n + 1:,}")
    print()
    print("Tokenizer : ggml-vocab-gemma-4.gguf (vocab_size=262144)")
    print("Source    : github.com/ggml-org/llama.cpp")
