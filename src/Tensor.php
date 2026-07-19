<?php

declare(strict_types=1);

namespace Llm;

require_once __DIR__ . '/Matrix.php';

/**
 * Chapter 4's Value engine, promoted from single numbers to whole matrices.
 *
 * Same three ideas, unchanged:
 *   1. every Tensor remembers which Tensors it was computed from,
 *   2. every operation knows its local derivative,
 *   3. backward() walks the graph in reverse, filling in gradients.
 *
 * What's new is granularity: one graph node now carries thousands of numbers,
 * and the local derivative rules become matrix equations. The payoff is speed —
 * the bookkeeping overhead is paid per *operation* instead of per *number*,
 * and the inner loops are tight float arithmetic that the JIT loves.
 *
 * This is exactly PyTorch's design: autograd over tensors.
 */
final class Tensor
{
    /** @var array<int, array<int, float>> gradient of the loss w.r.t. every entry, same shape as $data */
    public array $gradient;

    /** @var array<int, Tensor> */
    private array $parents = [];

    private ?\Closure $backwardFunction = null;

    /**
     * @param array<int, array<int, float>> $data rows of floats
     */
    public function __construct(
        public array $data,
    ) {
        $this->gradient = self::zeros(count($data), count($data[0]));
    }

    /** @return array<int, array<int, float>> */
    public static function zeros(int $rowCount, int $columnCount): array
    {
        return array_fill(0, $rowCount, array_fill(0, $columnCount, 0.0));
    }

    /** Random values in [-$scale, $scale) — how weights are born. */
    public static function random(int $rowCount, int $columnCount, RandomNumberGenerator $random, float $scale = 1.0): self
    {
        $data = [];
        for ($row = 0; $row < $rowCount; $row++) {
            for ($column = 0; $column < $columnCount; $column++) {
                $data[$row][$column] = ($random->nextFloat() * 2.0 - 1.0) * $scale;
            }
        }

        return new self($data);
    }

    public function rowCount(): int
    {
        return count($this->data);
    }

    public function columnCount(): int
    {
        return count($this->data[0]);
    }

    public function zeroGradient(): void
    {
        $this->gradient = self::zeros($this->rowCount(), $this->columnCount());
    }

    /** @return array<int, array<int, float>> */
    private static function transpose(array $matrix): array
    {
        $result = [];
        foreach ($matrix as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $result[$columnIndex][$rowIndex] = $value;
            }
        }

        return $result;
    }

    private static function addInto(array &$target, array $addend): void
    {
        foreach ($addend as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $target[$rowIndex][$columnIndex] += $value;
            }
        }
    }

    /**
     * Matrix multiplication — chapter 1's workhorse, now with gradients.
     *
     * The local rules are beautifully symmetric matrix equations:
     *   gradient of left  += resultGradient · rightᵀ
     *   gradient of right += leftᵀ · resultGradient
     * (Each is the matrix-shaped version of "multiply: the other factor".
     *  We don't derive them here — tests/check-ch5.php verifies them against
     *  numerical nudging, which is the argument that actually convinces.)
     */
    public function multiply(self $other): self
    {
        $result = new self(Matrix::multiply($this->data, $other->data));
        $result->parents = [$this, $other];
        $result->backwardFunction = function () use ($other, $result): void {
            self::addInto($this->gradient, Matrix::multiply($result->gradient, self::transpose($other->data)));
            self::addInto($other->gradient, Matrix::multiply(self::transpose($this->data), $result->gradient));
        };

        return $result;
    }

    /**
     * Add a 1×m row vector to every row — how a bias is applied to a whole batch.
     * Backward mirrors the broadcast: the bias soaked into every row, so its
     * gradient collects back from every row.
     */
    public function addRowVector(self $rowVector): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $data[$rowIndex][$columnIndex] = $value + $rowVector->data[0][$columnIndex];
            }
        }
        $result = new self($data);
        $result->parents = [$this, $rowVector];
        $result->backwardFunction = function () use ($rowVector, $result): void {
            self::addInto($this->gradient, $result->gradient);
            foreach ($result->gradient as $row) {
                foreach ($row as $columnIndex => $value) {
                    $rowVector->gradient[0][$columnIndex] += $value;
                }
            }
        };

        return $result;
    }

    /** Element-wise tanh, same rule as chapter 4 applied to every entry. */
    public function tanh(): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $data[$rowIndex][$columnIndex] = tanh($value);
            }
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            foreach ($result->gradient as $rowIndex => $row) {
                foreach ($row as $columnIndex => $upstream) {
                    $squashed = $result->data[$rowIndex][$columnIndex];
                    $this->gradient[$rowIndex][$columnIndex] += (1.0 - $squashed * $squashed) * $upstream;
                }
            }
        };

        return $result;
    }

    /**
     * Pick out rows by index: result row i = this row $rowIndices[i].
     *
     * This IS the embedding lookup. If each row of this tensor is a token's
     * learned vector, selectRows([13, 1, 20]) fetches the vectors for tokens
     * 13, 1, 20 in one shot. Backward scatters gradients onto the rows that
     * were used (a row selected twice accumulates twice — the += rule again).
     */
    public function selectRows(array $rowIndices): self
    {
        $data = [];
        foreach ($rowIndices as $i => $rowIndex) {
            $data[$i] = $this->data[$rowIndex];
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($rowIndices, $result): void {
            foreach ($rowIndices as $i => $rowIndex) {
                foreach ($result->gradient[$i] as $columnIndex => $value) {
                    $this->gradient[$rowIndex][$columnIndex] += $value;
                }
            }
        };

        return $result;
    }

    /**
     * The final step of every language model, fused into one operation:
     * softmax (scores → probabilities) + cross-entropy (chapter 3's loss).
     *
     * Row i of $this holds the model's raw scores ("logits") for every
     * vocabulary token; $targetIds[$i] says which token actually came next.
     * Returns a 1×1 tensor: the average negative log-probability of the truth.
     *
     * Fused because the combined gradient collapses to something almost
     * unbelievable:  gradient of logits = probabilities − one-hot(target).
     * "Predicted 40% on the right answer? Push its logit up by 0.6, push every
     * wrong logit down by its own probability." The two functions were made
     * for each other — every serious framework fuses them.
     *
     * @param array<int, int> $targetIds one correct token ID per row
     */
    public function softmaxCrossEntropy(array $targetIds): self
    {
        $rowCount = $this->rowCount();
        $probabilities = [];
        $totalLoss = 0.0;
        foreach ($this->data as $rowIndex => $logits) {
            $highest = max($logits); // subtracting the max avoids exp() overflow, changes nothing mathematically
            $exponentials = [];
            $sum = 0.0;
            foreach ($logits as $columnIndex => $logit) {
                $e = exp($logit - $highest);
                $exponentials[$columnIndex] = $e;
                $sum += $e;
            }
            foreach ($exponentials as $columnIndex => $e) {
                $probabilities[$rowIndex][$columnIndex] = $e / $sum;
            }
            $totalLoss += -log($probabilities[$rowIndex][$targetIds[$rowIndex]]);
        }

        $result = new self([[$totalLoss / $rowCount]]);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($probabilities, $targetIds, $rowCount, $result): void {
            $upstream = $result->gradient[0][0];
            foreach ($probabilities as $rowIndex => $row) {
                foreach ($row as $columnIndex => $probability) {
                    $oneHot = $columnIndex === $targetIds[$rowIndex] ? 1.0 : 0.0;
                    $this->gradient[$rowIndex][$columnIndex] += ($probability - $oneHot) / $rowCount * $upstream;
                }
            }
        };

        return $result;
    }

    /** Backpropagation — identical logic to Value::backward(), node by node. */
    public function backward(): void
    {
        $ordered = [];
        $visited = [];
        $visit = function (self $tensor) use (&$visit, &$ordered, &$visited): void {
            $objectId = spl_object_id($tensor);
            if (isset($visited[$objectId])) {
                return;
            }
            $visited[$objectId] = true;
            foreach ($tensor->parents as $parent) {
                $visit($parent);
            }
            $ordered[] = $tensor;
        };
        $visit($this);

        foreach ($this->gradient as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $this->gradient[$rowIndex][$columnIndex] = 1.0;
            }
        }
        foreach (array_reverse($ordered) as $tensor) {
            if ($tensor->backwardFunction !== null) {
                ($tensor->backwardFunction)();
            }
        }
    }
}
