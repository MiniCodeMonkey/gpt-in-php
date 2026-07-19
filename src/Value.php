<?php

declare(strict_types=1);

namespace Llm;

/**
 * The autograd engine — the mathematical heart of the entire course.
 *
 * A Value wraps one float, but unlike a plain float it REMEMBERS HOW IT WAS
 * MADE: which Values it was computed from and by which operation. Chaining
 * operations therefore builds a graph of the whole calculation.
 *
 * That memory buys us the superpower: after computing some final result
 * (the loss), calling ->backward() on it walks the graph in reverse and fills
 * every Value's ->gradient with the answer to:
 *
 *   "if this value were nudged up by a tiny amount,
 *    how much would the final result change?"
 *
 * Gradients for ALL inputs, in one backward pass, at roughly the cost of the
 * forward pass. PyTorch's autograd is this exact idea, industrialized.
 *
 * Each operation only knows its own LOCAL derivative (multiply, for example:
 * nudging one factor changes the product by the other factor). The chain rule
 * says: my gradient = my local derivative × the gradient of what I fed into.
 * Note the += everywhere: a Value used twice receives gradient from both uses.
 */
final class Value
{
    /** dLoss/dThis — how much the final result moves per tiny nudge of this value. */
    public float $gradient = 0.0;

    /** @var array<int, Value> the Values this one was computed from */
    private array $parents = [];

    /** @var \Closure|null pushes this Value's gradient back onto its parents */
    private ?\Closure $backwardFunction = null;

    public function __construct(
        public float $data,
    ) {
    }

    /** Convenience: lets every operation accept a plain float or a Value. */
    public static function of(self|float $x): self
    {
        return $x instanceof self ? $x : new self($x);
    }

    public function add(self|float $other): self
    {
        $other = self::of($other);
        $result = new self($this->data + $other->data);
        $result->parents = [$this, $other];
        $result->backwardFunction = function () use ($other, $result): void {
            // Local derivative of a sum is 1 for both sides: gradient flows straight through.
            $this->gradient += $result->gradient;
            $other->gradient += $result->gradient;
        };

        return $result;
    }

    public function subtract(self|float $other): self
    {
        $other = self::of($other);
        $result = new self($this->data - $other->data);
        $result->parents = [$this, $other];
        $result->backwardFunction = function () use ($other, $result): void {
            $this->gradient += $result->gradient;
            $other->gradient += -$result->gradient;
        };

        return $result;
    }

    public function multiply(self|float $other): self
    {
        $other = self::of($other);
        $result = new self($this->data * $other->data);
        $result->parents = [$this, $other];
        $result->backwardFunction = function () use ($other, $result): void {
            // Nudge one factor → the product moves by the OTHER factor's value.
            $this->gradient += $other->data * $result->gradient;
            $other->gradient += $this->data * $result->gradient;
        };

        return $result;
    }

    /** Raise to a constant power, e.g. ->power(2.0) to square. */
    public function power(float $exponent): self
    {
        $result = new self($this->data ** $exponent);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($exponent, $result): void {
            $this->gradient += $exponent * ($this->data ** ($exponent - 1.0)) * $result->gradient;
        };

        return $result;
    }

    public function exponential(): self
    {
        $result = new self(exp($this->data));
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            // e^x is its own derivative — which is the result we already computed.
            $this->gradient += $result->data * $result->gradient;
        };

        return $result;
    }

    public function logarithm(): self
    {
        $result = new self(log($this->data));
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            $this->gradient += (1.0 / $this->data) * $result->gradient;
        };

        return $result;
    }

    /**
     * The classic squashing activation: maps any number into (-1, 1).
     * This is what makes neural networks NON-linear — without it, stacking
     * layers would collapse into one big (useless) linear function.
     */
    public function tanh(): self
    {
        $squashed = tanh($this->data);
        $result = new self($squashed);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($squashed, $result): void {
            $this->gradient += (1.0 - $squashed * $squashed) * $result->gradient;
        };

        return $result;
    }

    /**
     * Backpropagation. Call this on the final result (the loss).
     *
     * Order matters: a node may only push gradient to its parents after ALL of
     * its own incoming gradient has arrived. A topological sort of the graph
     * guarantees that; then we sweep through it once, in reverse.
     */
    public function backward(): void
    {
        $ordered = [];
        $visited = [];
        $visit = function (self $value) use (&$visit, &$ordered, &$visited): void {
            $objectId = spl_object_id($value);
            if (isset($visited[$objectId])) {
                return;
            }
            $visited[$objectId] = true;
            foreach ($value->parents as $parent) {
                $visit($parent);
            }
            $ordered[] = $value;
        };
        $visit($this);

        $this->gradient = 1.0; // the loss moves 1-for-1 with itself
        foreach (array_reverse($ordered) as $value) {
            if ($value->backwardFunction !== null) {
                ($value->backwardFunction)();
            }
        }
    }
}
