# GPT, in PHP

A GPT-style language model built **completely from scratch in pure PHP** — no ML
libraries, no Composer dependencies, not even a matrix package. Tokenizers, an
autograd engine, tensors, self-attention, a full transformer, sampling, and a
two-stage-trained chatbot: about 1,500 lines of dependency-free PHP, with every
gradient verified against numerical differentiation.

Built as an interactive learning course — the code was written chapter by chapter
with [Claude Code](https://claude.com/claude-code), paired with an interactive
textbook with 15 live demos built from the real training runs.

**📖 Read the textbook: https://minicodemonkey.github.io/gpt-in-php/** — its final
demo runs the trained chatbot's full transformer forward pass right in the page.

## The results

Validation loss on 3,203 held-out names (lower is better; ln 27 ≈ 3.2958 is the
know-nothing baseline):

| model | chapter | parameters | val loss |
|---|---|---|---|
| bigram (counting) | 3 | 729 | 2.4546 |
| single attention head | 7 | 1,851 | 2.3267 |
| MLP language model | 6 | 6,899 | 2.1932 |
| GPT (2 blocks × 4 heads) | 8 | 27,355 | **2.1175** |

And the finale — a 29,496-parameter chatbot pretrained on a 5-person toy world,
then supervised-fine-tuned on dialogues, scoring 20/20 on its fact exam:

```
$ php bin/chat.php
you: where does ana live
bot: ana lives in the yellow house
you: who is rex
bot: rex is the dog of leo
```

## The curriculum

1. **The map** — what an LLM is; `Matrix::multiply`; benchmark your machine's FLOP/s
2. **Tokenization** — character tokenizer + a real byte-pair encoder
3. **The bigram model** — first generative model; negative log-likelihood loss
4. **Autograd** — `Value`: backpropagation from nothing, verified numerically
5. **Tensors** — the engine at matrix scale; softmax + cross-entropy; a neural
   bigram that rediscovers the counting table from noise
6. **MLP language model** — embeddings, context windows, train/val discipline
7. **Self-attention** — queries/keys/values, causal masking, position embeddings
8. **The Transformer** — residual streams, layer norm, multi-head, stacked blocks
9. **Inference** — temperature, top-k, and the KV-cache idea
10. **The chatbot** — word tokenizer, chat template, pretraining → SFT, `chat.php`
11. **To the frontier** — scaling laws, RLHF, what separates this from Claude

## Running it

Requires PHP ≥ 8.3 with OPcache. Everything runs from the repo root:

```sh
php tests/check-ch4.php          # per-chapter check suites (ch1–ch10)
./bin/phpj bin/train-gpt.php     # train the GPT (~6 min; phpj enables the JIT)
./bin/phpj bin/train-chatbot.php # pretrain + fine-tune the chatbot (~4 min)
php bin/chat.php                 # talk to it
php bin/generate.php --temperature=1.2 --prefix=ma --count=10
php bin/frontier-math.php        # how long GPT-3 would take on your machine
```

`bin/phpj` runs PHP with the JIT compiler and a bigger memory limit — about a 6×
speedup on the training hot loops. Everything is seeded and deterministic.

Trained weights for the GPT and chatbot are included in `models/`, so `chat.php`
and `generate.php` work without retraining.

## Credits

- `data/names.txt` (32,033 names) comes from Andrej Karpathy's
  [makemore](https://github.com/karpathy/makemore); the course's pedagogy owes a
  lot to his *Neural Networks: Zero to Hero* series.
- Architecture follows Vaswani et al., *Attention Is All You Need* (2017), in its
  GPT-2-style decoder-only form.
