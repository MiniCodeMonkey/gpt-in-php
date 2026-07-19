<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Tensor.php';
require_once __DIR__ . '/SelfAttentionHead.php';

/**
 * The smallest language model with self-attention at its core:
 *
 *   token IDs → token embeddings + POSITION embeddings → one attention head → logits
 *
 * Position embeddings are new, and attention is why: the attention mechanism
 * itself has no idea WHERE anything is — it sees a bag of vectors and compares
 * contents. (The MLP got position for free from its fixed slots.) So we give
 * each position 0…T−1 its own learned vector and add it to the token's vector:
 * now "m at position 1" and "m at position 4" look different, and attention
 * can learn position-aware behavior.
 *
 * Unlike the MLP, one forward pass trains ALL positions at once: feeding
 * [·, e, m, m, a] produces T predictions in parallel — position 0 predicts e,
 * position 1 predicts m, … — with the causal mask keeping each one honest.
 * One sequence = T training examples for the price of one pass; this
 * parallelism is why transformers train efficiently at scale.
 */
final class AttentionLanguageModel
{
    public Tensor $tokenEmbeddings;
    public Tensor $positionEmbeddings;
    public SelfAttentionHead $attentionHead;
    public Tensor $outputWeights;
    public Tensor $outputBias;

    public function __construct(
        public readonly int $vocabularySize,
        public readonly int $contextLength,
        int $embeddingDimensions,
        RandomNumberGenerator $random,
    ) {
        $this->tokenEmbeddings = Tensor::random($vocabularySize, $embeddingDimensions, $random, 0.6);
        $this->positionEmbeddings = Tensor::random($contextLength, $embeddingDimensions, $random, 0.6);
        $this->attentionHead = new SelfAttentionHead($embeddingDimensions, $embeddingDimensions, $random);
        $this->outputWeights = Tensor::random($embeddingDimensions, $vocabularySize, $random, 1.0 / sqrt($embeddingDimensions));
        $this->outputBias = new Tensor([array_fill(0, $vocabularySize, 0.0)]);
    }

    /**
     * @param array<int, int> $tokenIds the sequence so far (length T ≤ contextLength)
     * @return Tensor T×vocabularySize — one next-token prediction PER position
     */
    public function computeLogits(array $tokenIds): Tensor
    {
        $positions = range(0, count($tokenIds) - 1);

        $embedded = $this->tokenEmbeddings->selectRows($tokenIds)
            ->addElementwise($this->positionEmbeddings->selectRows($positions));

        return $this->attentionHead->forward($embedded)
            ->multiply($this->outputWeights)
            ->addRowVector($this->outputBias);
    }

    /**
     * @param array<int, int> $inputIds sequence positions 0…T−1
     * @param array<int, int> $targetIds the token that follows each position
     */
    public function computeLoss(array $inputIds, array $targetIds): Tensor
    {
        return $this->computeLogits($inputIds)->softmaxCrossEntropy($targetIds);
    }

    /** @return array<int, Tensor> */
    public function parameters(): array
    {
        return [
            $this->tokenEmbeddings,
            $this->positionEmbeddings,
            ...$this->attentionHead->parameters(),
            $this->outputWeights,
            $this->outputBias,
        ];
    }

    public function parameterCount(): int
    {
        $count = 0;
        foreach ($this->parameters() as $parameter) {
            $count += $parameter->rowCount() * $parameter->columnCount();
        }

        return $count;
    }

    public function zeroGradients(): void
    {
        foreach ($this->parameters() as $parameter) {
            $parameter->zeroGradient();
        }
    }

    public function applyGradientStep(float $learningRate): void
    {
        foreach ($this->parameters() as $parameter) {
            foreach ($parameter->gradient as $rowIndex => $row) {
                foreach ($row as $columnIndex => $gradient) {
                    $parameter->data[$rowIndex][$columnIndex] -= $learningRate * $gradient;
                }
            }
        }
    }

    /** Generate one name: grow the sequence, always predicting from the LAST position's logits. */
    public function generate(RandomNumberGenerator $random, CharacterTokenizer $tokenizer, int $maximumLength = 24): string
    {
        $tokenIds = [0];
        $name = '';
        for ($length = 0; $length < $maximumLength; $length++) {
            $window = array_slice($tokenIds, -$this->contextLength);
            $logits = $this->computeLogits($window)->data[count($window) - 1];

            $highest = max($logits);
            $exponentials = array_map(fn (float $logit) => exp($logit - $highest), $logits);
            $sum = array_sum($exponentials);
            $probabilities = array_map(fn (float $e) => $e / $sum, $exponentials);

            $nextTokenId = $random->sampleFromDistribution($probabilities);
            if ($nextTokenId === 0) {
                break;
            }
            $name .= $tokenizer->vocabulary()[$nextTokenId];
            $tokenIds[] = $nextTokenId;
        }

        return $name;
    }
}
