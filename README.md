# GPT, in PHP

Building a GPT-style language model completely from scratch in pure PHP — no ML
libraries, no dependencies — as an interactive course. Ends with a tiny chatbot
(`php chat.php`) and the theory connecting it to frontier models.

The course companion is an interactive web textbook (source in `docs/textbook.html`,
published as a claude.ai artifact — link in the session / memory notes).

## Layout

- `src/` — the model, built up chapter by chapter (all code written by Claude, studied by Mathias)
- `bin/` — runnable scripts (benchmarks, training, generation)
- `bin/phpj` — runs PHP with the JIT compiler enabled; use for anything heavy
- `tests/` — per-chapter check scripts: `php tests/check-ch1.php`
- `data/` — training datasets (added as chapters need them)
- `docs/` — the textbook source and design notes

## Requirements

PHP ≥ 8.3 with OPcache (using 8.5.8 via Laravel Herd).

## Progress

- [x] Ch 1 — The map (Matrix::multiply + FLOP/s benchmark)
- [x] Ch 2 — Tokenization (CharacterTokenizer + BytePairEncoder)
- [x] Ch 3 — Bigram model (BigramModel + RandomNumberGenerator, loss 2.4546)
- [x] Ch 4 — Autograd (Value engine + TinyNeuralNetwork learns XOR)
- [ ] Ch 5 — Tensors
- [ ] Ch 6 — MLP language model
- [ ] Ch 7 — Self-attention
- [ ] Ch 8 — The Transformer
- [ ] Ch 9 — Inference
- [ ] Ch 10 — The chatbot
- [ ] Ch 11 — To the frontier
