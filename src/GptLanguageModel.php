<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Tensor.php';
require_once __DIR__ . '/TransformerBlock.php';

/**
 * The GPT. Architecturally, this IS GPT-2 — same blueprint, tiny numbers:
 *
 *   token IDs
 *     → token embeddings + position embeddings      (chapters 5-7)
 *     → N transformer blocks                        (chapter 8)
 *     → final layer normalization
 *     → projection to vocabulary logits             (chapter 5)
 *
 * Everything below is machinery you have already built, assembled.
 */
final class GptLanguageModel
{
    public Tensor $tokenEmbeddings;
    public Tensor $positionEmbeddings;
    /** @var array<int, TransformerBlock> */
    public array $blocks = [];
    public Tensor $finalNormGain;
    public Tensor $finalNormBias;
    public Tensor $outputWeights;
    public Tensor $outputBias;

    public function __construct(
        public readonly int $vocabularySize,
        public readonly int $contextLength,
        public readonly int $embeddingDimensions,
        public readonly int $headCount,
        public readonly int $blockCount,
        RandomNumberGenerator $random,
    ) {
        $this->tokenEmbeddings = Tensor::random($vocabularySize, $embeddingDimensions, $random, 0.4);
        $this->positionEmbeddings = Tensor::random($contextLength, $embeddingDimensions, $random, 0.4);
        for ($i = 0; $i < $blockCount; $i++) {
            $this->blocks[] = new TransformerBlock($embeddingDimensions, $headCount, $random);
        }
        $this->finalNormGain = new Tensor([array_fill(0, $embeddingDimensions, 1.0)]);
        $this->finalNormBias = new Tensor([array_fill(0, $embeddingDimensions, 0.0)]);
        $this->outputWeights = Tensor::random($embeddingDimensions, $vocabularySize, $random, 1.0 / sqrt($embeddingDimensions));
        $this->outputBias = new Tensor([array_fill(0, $vocabularySize, 0.0)]);
    }

    /**
     * @param array<int, int> $tokenIds sequence of length T ≤ contextLength
     * @return Tensor T×vocabularySize — a next-token prediction per position
     */
    public function computeLogits(array $tokenIds): Tensor
    {
        $stream = $this->tokenEmbeddings->selectRows($tokenIds)
            ->addElementwise($this->positionEmbeddings->selectRows(range(0, count($tokenIds) - 1)));

        foreach ($this->blocks as $block) {
            $stream = $block->forward($stream);
        }

        return $stream
            ->layerNormalizeRows($this->finalNormGain, $this->finalNormBias)
            ->multiply($this->outputWeights)
            ->addRowVector($this->outputBias);
    }

    /**
     * @param array<int, int> $inputIds
     * @param array<int, int> $targetIds
     */
    public function computeLoss(array $inputIds, array $targetIds): Tensor
    {
        return $this->computeLogits($inputIds)->softmaxCrossEntropy($targetIds);
    }

    /** @return array<int, float> raw scores for the token that follows the sequence — inference's raw material */
    public function nextTokenLogits(array $tokenIds): array
    {
        return $this->computeLogits($tokenIds)->data[count($tokenIds) - 1];
    }

    /** @return array<int, float> probability of each token following the given sequence */
    public function nextTokenProbabilities(array $tokenIds): array
    {
        $logits = $this->nextTokenLogits($tokenIds);
        $highest = max($logits);
        $exponentials = array_map(fn (float $logit) => exp($logit - $highest), $logits);
        $sum = array_sum($exponentials);

        return array_map(fn (float $e) => $e / $sum, $exponentials);
    }

    public function generate(RandomNumberGenerator $random, CharacterTokenizer $tokenizer, int $maximumLength = 24): string
    {
        $tokenIds = [0];
        $name = '';
        for ($length = 0; $length < $maximumLength; $length++) {
            $window = array_slice($tokenIds, -$this->contextLength);
            $nextTokenId = $random->sampleFromDistribution($this->nextTokenProbabilities($window));
            if ($nextTokenId === 0) {
                break;
            }
            $name .= $tokenizer->vocabulary()[$nextTokenId];
            $tokenIds[] = $nextTokenId;
        }

        return $name;
    }

    /** @return array<int, Tensor> */
    public function parameters(): array
    {
        $parameters = [$this->tokenEmbeddings, $this->positionEmbeddings];
        foreach ($this->blocks as $block) {
            $parameters = [...$parameters, ...$block->parameters()];
        }

        return [...$parameters, $this->finalNormGain, $this->finalNormBias, $this->outputWeights, $this->outputBias];
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

    // ---- Persistence: minutes of training shouldn't be repeated ------------

    public function saveToFile(string $path): void
    {
        $weights = array_map(fn (Tensor $parameter) => $parameter->data, $this->parameters());
        file_put_contents($path, json_encode([
            'configuration' => [
                'vocabularySize' => $this->vocabularySize,
                'contextLength' => $this->contextLength,
                'embeddingDimensions' => $this->embeddingDimensions,
                'headCount' => $this->headCount,
                'blockCount' => $this->blockCount,
            ],
            'weights' => $weights,
        ]));
    }

    public static function loadFromFile(string $path): self
    {
        $saved = json_decode(file_get_contents($path), true);
        $configuration = $saved['configuration'];

        // Seed irrelevant: every weight is about to be overwritten.
        $model = new self(
            $configuration['vocabularySize'],
            $configuration['contextLength'],
            $configuration['embeddingDimensions'],
            $configuration['headCount'],
            $configuration['blockCount'],
            new RandomNumberGenerator(1),
        );
        foreach ($model->parameters() as $index => $parameter) {
            $parameter->data = $saved['weights'][$index];
        }

        return $model;
    }
}
