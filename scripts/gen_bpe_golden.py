#!/usr/bin/env python3
# Offline oracle. Run with AtlasLM/.venv (tokenizers 0.23.1). NOT in runtime path.
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
OUT = os.path.dirname(__file__) + "/../tests/bpe"
os.makedirs(OUT + "/golden", exist_ok=True)

# (a) pre-tokenizer golden: INPUT<TAB>byte-mapped-piece|byte-mapped-piece...
bl = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
cases = ["hello world", "café π", "½x", "don't  stop", "a\nb\tc",
         "  leading", "trailing  ", "MixedCASE123", "e'er 've 'll",
         "\n\n\n", "123456", "a1b2", "Ń óó", "tab\tafter"]
with open(OUT + "/golden/pretok_cases.tsv", "w", encoding="utf-8") as f:
    for s in cases:
        pieces = [p for p, _ in bl.pre_tokenize_str(s)]
        # store INPUT and pieces as \xNN-escaped so the .tsv is pure ASCII/loadable
        def esc(x): return "".join("\\x%02x" % b for b in x.encode("utf-8"))
        f.write(esc(s) + "\t" + "|".join(esc(p) for p in pieces) + "\n")

# (b) tiny corpus + golden tokenizer.json (small vocab for a fast exact gate).
# HARDENED: blank lines, leading/trailing spaces, an overlap run (wwwww),
# non-ASCII (café π ½), and contractions — so the golden catches the per-line
# newline rule, the overlap deltas, raw-split-then-map, and the ' matcher.
tiny = ("the quick brown fox\n"
        "jumps over the lazy dog\n"
        "\n"                              # blank line (must NOT create Ċ-fused tokens)
        "\n"                              # 2nd consecutive blank line: "...dog\n\n\nthe..."
                                          # is a \n\n\n region. Whole-buffer splitting fuses
                                          # this into a "ĊĊ" token because \s+(?!\S) greedily
                                          # spans the line boundary; per-line splitting must
                                          # NOT produce that token.
        "the the the quick quick\n"
        "  indented start\n"             # leading spaces
        "trailing end  \n"               # trailing spaces
        "don't stop, e'er 've 'll\n"     # contractions
        "wwwww aaa abab ababab\n"        # overlap runs
        "café π ½ Ń\n") * 40             # non-ASCII
with open(OUT + "/tiny_corpus.txt", "w", encoding="utf-8") as f:
    f.write(tiny)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.post_processor = processors.ByteLevel(trim_offsets=False)
tok.train([OUT + "/tiny_corpus.txt"],
          trainers.BpeTrainer(vocab_size=320,
                              special_tokens=["<PAD>","<UNK>","<BOS>","<EOS>"]))
tok.save(OUT + "/golden/tiny_tokenizer.json")
print("wrote fixtures to", OUT)
