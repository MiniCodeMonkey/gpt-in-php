<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Value.php';

/**
 * The smallest real neural network, built on the Value autograd engine.
 *
 * A Neuron is: multiply each input by a learned weight, add them up, add a
 * learned bias, squash with tanh. A Layer is neurons side by side. A network
 * is layers in sequence. That's it — a transformer is this idea, arranged
 * cleverly and scaled up.
 *
 * (Chapter 4 uses one scalar Value per number, which is beautifully clear and
 *  hopelessly slow. Chapter 5 rebuilds the same math with matrices for speed.)
 */
final class Neuron
{
    /** @var array<int, Value> */
    private array $weights = [];
    private Value $bias;

    public function __construct(int $inputCount, RandomNumberGenerator $random)
    {
        for ($i = 0; $i < $inputCount; $i++) {
            $this->weights[] = new Value($random->nextFloat() * 2.0 - 1.0); // start random in [-1, 1)
        }
        $this->bias = new Value(0.0);
    }

    /**
     * @param array<int, Value> $inputs
     */
    public function forward(array $inputs): Value
    {
        $sum = $this->bias;
        foreach ($this->weights as $index => $weight) {
            $sum = $sum->add($weight->multiply($inputs[$index]));
        }

        return $sum->tanh();
    }

    /** @return array<int, Value> every learnable knob in this neuron */
    public function parameters(): array
    {
        return [...$this->weights, $this->bias];
    }
}

final class Layer
{
    /** @var array<int, Neuron> */
    private array $neurons = [];

    public function __construct(int $inputCount, int $neuronCount, RandomNumberGenerator $random)
    {
        for ($i = 0; $i < $neuronCount; $i++) {
            $this->neurons[] = new Neuron($inputCount, $random);
        }
    }

    /**
     * @param array<int, Value> $inputs
     * @return array<int, Value> one output per neuron
     */
    public function forward(array $inputs): array
    {
        return array_map(fn (Neuron $neuron) => $neuron->forward($inputs), $this->neurons);
    }

    /** @return array<int, Value> */
    public function parameters(): array
    {
        return array_merge(...array_map(fn (Neuron $neuron) => $neuron->parameters(), $this->neurons));
    }
}

final class TinyNeuralNetwork
{
    /** @var array<int, Layer> */
    private array $layers = [];

    /**
     * @param array<int, int> $layerSizes e.g. [2, 4, 1]: 2 inputs → 4 hidden neurons → 1 output
     */
    public function __construct(array $layerSizes, RandomNumberGenerator $random)
    {
        for ($i = 0; $i < count($layerSizes) - 1; $i++) {
            $this->layers[] = new Layer($layerSizes[$i], $layerSizes[$i + 1], $random);
        }
    }

    /**
     * @param array<int, float> $inputs
     */
    public function forward(array $inputs): Value
    {
        $activations = array_map(fn (float $x) => new Value($x), $inputs);
        foreach ($this->layers as $layer) {
            $activations = $layer->forward($activations);
        }

        return $activations[0]; // our networks here end in a single output neuron
    }

    /** @return array<int, Value> */
    public function parameters(): array
    {
        return array_merge(...array_map(fn (Layer $layer) => $layer->parameters(), $this->layers));
    }

    /** Gradients accumulate with +=, so they must be reset before each backward pass. */
    public function zeroGradients(): void
    {
        foreach ($this->parameters() as $parameter) {
            $parameter->gradient = 0.0;
        }
    }
}
