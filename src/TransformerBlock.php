<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Tensor.php';
require_once __DIR__ . '/SelfAttentionHead.php';

/**
 * One transformer block — the unit that gets stacked N times to make a GPT.
 * (GPT-2: 12 of these. GPT-3: 96. Ours: 2. Same blueprint.)
 *
 * Two sub-layers, each wrapped the same way (normalize → compute → add back):
 *
 *   1. Multi-head attention — COMMUNICATION. Positions exchange information.
 *      Several heads run in parallel, each free to learn a different lookup
 *      (one might track the previous character, another the start of the
 *      name); their outputs are concatenated and mixed by a projection.
 *   2. Feed-forward network — COMPUTATION. Each position, alone, processes
 *      what it just gathered through a small 2-layer neural net (widened 4×
 *      in the middle — room to think).
 *
 * The wrapping is the deep-network survival kit:
 *
 *   · Residual connections: each sub-layer's output is ADDED to its input,
 *     never replacing it. Think of one stream of vectors flowing upward —
 *     the "residual stream" — with every sub-layer reading from it and
 *     writing small additive edits back. Because addition passes gradients
 *     through untouched (chapter 4's add rule), the loss signal flows from
 *     the top of a deep stack straight down to the bottom. Without this,
 *     deep networks simply don't train.
 *   · Layer normalization before each sub-layer, keeping a deep stack's
 *     activations in every layer's responsive range.
 */
final class TransformerBlock
{
    /** @var array<int, SelfAttentionHead> */
    private array $attentionHeads = [];
    private Tensor $attentionProjection;

    private Tensor $feedForwardExpandWeights;
    private Tensor $feedForwardExpandBias;
    private Tensor $feedForwardContractWeights;
    private Tensor $feedForwardContractBias;

    private Tensor $attentionNormGain;
    private Tensor $attentionNormBias;
    private Tensor $feedForwardNormGain;
    private Tensor $feedForwardNormBias;

    public function __construct(int $embeddingDimensions, int $headCount, RandomNumberGenerator $random)
    {
        $headSize = intdiv($embeddingDimensions, $headCount);
        for ($i = 0; $i < $headCount; $i++) {
            $this->attentionHeads[] = new SelfAttentionHead($embeddingDimensions, $headSize, $random);
        }
        $this->attentionProjection = Tensor::random($embeddingDimensions, $embeddingDimensions, $random, 1.0 / sqrt($embeddingDimensions));

        $widened = 4 * $embeddingDimensions; // room to think
        $this->feedForwardExpandWeights = Tensor::random($embeddingDimensions, $widened, $random, 1.0 / sqrt($embeddingDimensions));
        $this->feedForwardExpandBias = new Tensor([array_fill(0, $widened, 0.0)]);
        $this->feedForwardContractWeights = Tensor::random($widened, $embeddingDimensions, $random, 1.0 / sqrt($widened));
        $this->feedForwardContractBias = new Tensor([array_fill(0, $embeddingDimensions, 0.0)]);

        $this->attentionNormGain = new Tensor([array_fill(0, $embeddingDimensions, 1.0)]);
        $this->attentionNormBias = new Tensor([array_fill(0, $embeddingDimensions, 0.0)]);
        $this->feedForwardNormGain = new Tensor([array_fill(0, $embeddingDimensions, 1.0)]);
        $this->feedForwardNormBias = new Tensor([array_fill(0, $embeddingDimensions, 0.0)]);
    }

    public function forward(Tensor $stream): Tensor
    {
        // Communication: normalize, attend with every head, mix, add back.
        $normalized = $stream->layerNormalizeRows($this->attentionNormGain, $this->attentionNormBias);
        $headOutputs = array_map(fn (SelfAttentionHead $head) => $head->forward($normalized), $this->attentionHeads);
        $attended = Tensor::concatenateColumns($headOutputs)->multiply($this->attentionProjection);
        $stream = $stream->addElementwise($attended);

        // Computation: normalize, think per-position, add back.
        $normalized = $stream->layerNormalizeRows($this->feedForwardNormGain, $this->feedForwardNormBias);
        $thought = $normalized
            ->multiply($this->feedForwardExpandWeights)
            ->addRowVector($this->feedForwardExpandBias)
            ->tanh()
            ->multiply($this->feedForwardContractWeights)
            ->addRowVector($this->feedForwardContractBias);

        return $stream->addElementwise($thought);
    }

    /** @return array<int, SelfAttentionHead> for attention-pattern inspection */
    public function attentionHeads(): array
    {
        return $this->attentionHeads;
    }

    /** @return array<int, Tensor> */
    public function parameters(): array
    {
        $parameters = [];
        foreach ($this->attentionHeads as $head) {
            $parameters = [...$parameters, ...$head->parameters()];
        }

        return [
            ...$parameters,
            $this->attentionProjection,
            $this->feedForwardExpandWeights,
            $this->feedForwardExpandBias,
            $this->feedForwardContractWeights,
            $this->feedForwardContractBias,
            $this->attentionNormGain,
            $this->attentionNormBias,
            $this->feedForwardNormGain,
            $this->feedForwardNormBias,
        ];
    }
}
