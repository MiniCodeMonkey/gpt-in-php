<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Tensor.php';

/**
 * The first model in the course that can beat the bigram: a multi-layer
 * perceptron language model (the architecture from Bengio et al., 2003 —
 * the paper that started neural language modeling).
 *
 * The bigram's fatal limit was one character of context. This model reads a
 * WINDOW of characters, and sidesteps the exploding-table problem with
 * embeddings:
 *
 *   1. Each character becomes a small learned vector (its embedding) — not a
 *      row in a 27^n table. Characters that behave alike end up with similar
 *      vectors, so what the model learns about "ka" transfers to "kо́a-like"
 *      contexts it has never seen. THIS is generalization, the thing counting
 *      could never do.
 *   2. The window's vectors are concatenated and fed through a hidden layer
 *      of neurons (chapter 4's tanh neurons, matrix form) that learn to
 *      detect useful patterns in the context.
 *   3. A final layer converts the hidden activations into 27 logits.
 *
 * Pipeline per batch:
 *   contexts → embed+concat → ·hiddenWeights+bias → tanh → ·outputWeights+bias → logits
 */
final class MultiLayerPerceptronLanguageModel
{
    public Tensor $embeddingTable;
    public Tensor $hiddenWeights;
    public Tensor $hiddenBias;
    public Tensor $outputWeights;
    public Tensor $outputBias;

    public function __construct(
        public readonly int $vocabularySize,
        public readonly int $contextLength,
        int $embeddingDimensions,
        int $hiddenNeuronCount,
        RandomNumberGenerator $random,
    ) {
        $inputWidth = $contextLength * $embeddingDimensions;

        // Weight scale ~ 1/sqrt(inputs per neuron): keeps early activations in
        // tanh's living zone instead of its flat saturated tails, where the
        // gradient (1 - tanh²) is ~0 and learning stalls. A real-world lesson:
        // bad initialization can kill a network before training begins.
        $this->embeddingTable = Tensor::random($vocabularySize, $embeddingDimensions, $random, 1.0);
        $this->hiddenWeights = Tensor::random($inputWidth, $hiddenNeuronCount, $random, 1.0 / sqrt($inputWidth));
        $this->hiddenBias = new Tensor([array_fill(0, $hiddenNeuronCount, 0.0)]);
        $this->outputWeights = Tensor::random($hiddenNeuronCount, $vocabularySize, $random, 1.0 / sqrt($hiddenNeuronCount));
        $this->outputBias = new Tensor([array_fill(0, $vocabularySize, 0.0)]);
    }

    /**
     * @param array<int, array<int, int>> $contexts one window of token IDs per example
     */
    public function computeLogits(array $contexts): Tensor
    {
        return $this->embeddingTable
            ->selectAndConcatenateRows($contexts)
            ->multiply($this->hiddenWeights)
            ->addRowVector($this->hiddenBias)
            ->tanh()
            ->multiply($this->outputWeights)
            ->addRowVector($this->outputBias);
    }

    /**
     * @param array<int, array<int, int>> $contexts
     * @param array<int, int> $targetIds
     */
    public function computeLoss(array $contexts, array $targetIds): Tensor
    {
        return $this->computeLogits($contexts)->softmaxCrossEntropy($targetIds);
    }

    /** @return array<int, Tensor> */
    public function parameters(): array
    {
        return [$this->embeddingTable, $this->hiddenWeights, $this->hiddenBias, $this->outputWeights, $this->outputBias];
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

    /** One plain gradient-descent step over every parameter. */
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

    /**
     * Generate one name: start with an all-boundary window, predict, sample,
     * slide the window one character to the left, repeat. Same loop as ever —
     * the model just sees more than one character now.
     */
    public function generate(RandomNumberGenerator $random, CharacterTokenizer $tokenizer, int $maximumLength = 24): string
    {
        $window = array_fill(0, $this->contextLength, 0);
        $name = '';
        for ($length = 0; $length < $maximumLength; $length++) {
            $logits = $this->computeLogits([$window])->data[0];
            $highest = max($logits);
            $exponentials = array_map(fn (float $logit) => exp($logit - $highest), $logits);
            $sum = array_sum($exponentials);
            $probabilities = array_map(fn (float $e) => $e / $sum, $exponentials);

            $nextTokenId = $random->sampleFromDistribution($probabilities);
            if ($nextTokenId === 0) {
                break;
            }
            $name .= $tokenizer->vocabulary()[$nextTokenId];
            $window = [...array_slice($window, 1), $nextTokenId];
        }

        return $name;
    }
}
