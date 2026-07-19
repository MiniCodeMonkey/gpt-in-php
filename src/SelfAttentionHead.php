<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Tensor.php';

/**
 * One head of self-attention — the 2017 idea ("Attention Is All You Need")
 * that replaced fixed context windows and made GPT possible.
 *
 * The MLP's flaw: a rigid window. Position 7 sees exactly positions 4-6,
 * mashed together in fixed slots. Attention's fix: let EVERY position look at
 * EVERY earlier position and decide, based on content, how much each one
 * matters right now.
 *
 * The mechanism is a soft lookup, in three learned projections of each
 * position's vector:
 *
 *   query — "what am I looking for?"     (this position's search terms)
 *   key   — "what do I contain?"         (how this position advertises itself)
 *   value — "what will I hand over?"     (the actual payload, if attended to)
 *
 * Every query is dotted against every key → a T×T grid of match scores.
 * Scores are scaled by 1/√headSize (dot products of longer vectors are
 * naturally bigger; unscaled, softmax saturates and gradients die), masked so
 * the future is invisible, and softmaxed into attention weights: each position
 * now holds a probability distribution over "whose value do I take?". The
 * output is each position's attention-weighted blend of the values.
 *
 * Every step is differentiable, so gradient descent LEARNS what to look for —
 * nobody hand-codes the lookup patterns.
 */
final class SelfAttentionHead
{
    public Tensor $queryWeights;
    public Tensor $keyWeights;
    public Tensor $valueWeights;

    /** @var array<int, array<int, float>>|null the last T×T attention pattern — kept for inspection/visualization */
    public ?array $lastAttentionWeights = null;

    public function __construct(
        int $embeddingDimensions,
        public readonly int $headSize,
        RandomNumberGenerator $random,
    ) {
        $scale = 1.0 / sqrt($embeddingDimensions);
        $this->queryWeights = Tensor::random($embeddingDimensions, $headSize, $random, $scale);
        $this->keyWeights = Tensor::random($embeddingDimensions, $headSize, $random, $scale);
        $this->valueWeights = Tensor::random($embeddingDimensions, $headSize, $random, $scale);
    }

    /**
     * @param Tensor $sequence T×embeddingDimensions — one vector per position
     * @return Tensor T×headSize — each position's blend of the values it attended to
     */
    public function forward(Tensor $sequence): Tensor
    {
        $queries = $sequence->multiply($this->queryWeights);
        $keys = $sequence->multiply($this->keyWeights);
        $values = $sequence->multiply($this->valueWeights);

        $attentionWeights = $queries
            ->multiply($keys->transposed())               // every query · every key → T×T match scores
            ->multiplyByScalar(1.0 / sqrt($this->headSize))
            ->maskFuturePositions()
            ->softmaxRows();                              // each row: a distribution over visible positions

        $this->lastAttentionWeights = $attentionWeights->data;

        return $attentionWeights->multiply($values);
    }

    /** @return array<int, Tensor> */
    public function parameters(): array
    {
        return [$this->queryWeights, $this->keyWeights, $this->valueWeights];
    }
}
