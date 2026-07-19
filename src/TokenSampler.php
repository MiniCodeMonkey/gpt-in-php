<?php

declare(strict_types=1);

namespace Llm;

/**
 * The inference knobs. Training froze the model; every "personality" setting
 * you've seen on an LLM API happens HERE, in the last inch between logits and
 * the chosen token.
 *
 * Temperature — divide all logits by T before softmax:
 *   T → 0   sharpens to certainty: always the single most likely token (greedy).
 *   T = 1   the model's honest learned distribution.
 *   T > 1   flattens toward uniform: adventurous, then unhinged.
 * (Same exponential mechanics as physical temperature in statistical physics —
 *  the name is not a metaphor.)
 *
 * Top-k — before sampling, keep only the k highest-scoring tokens and
 * renormalize. With big vocabularies the model puts a sliver of probability on
 * thousands of nonsense tokens; individually negligible, together they're a
 * real chance of derailment per token. Top-k amputates that tail.
 */
final class TokenSampler
{
    public function __construct(
        private readonly float $temperature = 1.0,
        private readonly ?int $keepTopK = null,
    ) {
    }

    /**
     * @param array<int, float> $logits raw scores, one per vocabulary token
     */
    public function sampleFromLogits(array $logits, RandomNumberGenerator $random): int
    {
        // Temperature 0 means "no randomness at all": take the argmax.
        if ($this->temperature < 1e-6) {
            return array_search(max($logits), $logits, true);
        }

        $scaled = array_map(fn (float $logit) => $logit / $this->temperature, $logits);

        if ($this->keepTopK !== null) {
            $ranked = $scaled;
            rsort($ranked);
            $cutoff = $ranked[$this->keepTopK - 1];
            foreach ($scaled as $tokenId => $value) {
                if ($value < $cutoff) {
                    $scaled[$tokenId] = -1e9; // exiled from the distribution
                }
            }
        }

        $highest = max($scaled);
        $exponentials = array_map(fn (float $value) => exp($value - $highest), $scaled);
        $sum = array_sum($exponentials);

        return $random->sampleFromDistribution(array_map(fn (float $e) => $e / $sum, $exponentials));
    }
}
