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
     * The chapter 6 upgrade of selectRows: fetch SEVERAL rows per example and
     * lay them side by side. For context window [5, 13, 13] ("emm") and a
     * 27×8 embedding table, the result row is 24 numbers: e's vector, then
     * m's, then m's again — a whole context packed into one flat row, ready
     * to feed a neural network. Backward scatters each slice back onto the
     * embedding row it came from (m's row collects from both uses).
     *
     * @param array<int, array<int, int>> $rowIndexGroups one group of row indices per example
     */
    public function selectAndConcatenateRows(array $rowIndexGroups): self
    {
        $columnCount = $this->columnCount();
        $data = [];
        foreach ($rowIndexGroups as $i => $rowIndices) {
            $flat = [];
            foreach ($rowIndices as $rowIndex) {
                foreach ($this->data[$rowIndex] as $value) {
                    $flat[] = $value;
                }
            }
            $data[$i] = $flat;
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($rowIndexGroups, $columnCount, $result): void {
            foreach ($rowIndexGroups as $i => $rowIndices) {
                foreach ($rowIndices as $position => $rowIndex) {
                    $offset = $position * $columnCount;
                    for ($column = 0; $column < $columnCount; $column++) {
                        $this->gradient[$rowIndex][$column] += $result->gradient[$i][$offset + $column];
                    }
                }
            }
        };

        return $result;
    }

    /** Differentiable transpose — needed for attention's queries · keysᵀ. */
    public function transposed(): self
    {
        $result = new self(self::transpose($this->data));
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            self::addInto($this->gradient, self::transpose($result->gradient));
        };

        return $result;
    }

    /** Multiply every entry by one constant (e.g. attention's 1/√headSize). */
    public function multiplyByScalar(float $scalar): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $data[$rowIndex][$columnIndex] = $value * $scalar;
            }
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($scalar, $result): void {
            foreach ($result->gradient as $rowIndex => $row) {
                foreach ($row as $columnIndex => $value) {
                    $this->gradient[$rowIndex][$columnIndex] += $value * $scalar;
                }
            }
        };

        return $result;
    }

    /** Element-wise addition of two same-shaped tensors (residual connections, position embeddings). */
    public function addElementwise(self $other): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $data[$rowIndex][$columnIndex] = $value + $other->data[$rowIndex][$columnIndex];
            }
        }
        $result = new self($data);
        $result->parents = [$this, $other];
        $result->backwardFunction = function () use ($other, $result): void {
            self::addInto($this->gradient, $result->gradient);
            self::addInto($other->gradient, $result->gradient);
        };

        return $result;
    }

    /**
     * The causality gate. In a T×T score matrix where row t holds position t's
     * interest in every position, entries above the diagonal are interest in
     * the FUTURE — position 2 peeking at position 5. Generation happens left
     * to right, so the future must be invisible during training too (or the
     * model learns to cheat and falls apart the moment it has to generate).
     *
     * Setting those scores to -1e9 makes the upcoming softmax give them
     * probability ~zero. Backward: blocked cells pass no gradient.
     */
    public function maskFuturePositions(): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $row) {
            foreach ($row as $columnIndex => $value) {
                $data[$rowIndex][$columnIndex] = $columnIndex > $rowIndex ? -1e9 : $value;
            }
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            foreach ($result->gradient as $rowIndex => $row) {
                foreach ($row as $columnIndex => $value) {
                    if ($columnIndex <= $rowIndex) {
                        $this->gradient[$rowIndex][$columnIndex] += $value;
                    }
                }
            }
        };

        return $result;
    }

    /**
     * Softmax applied to every row independently — turns each row of attention
     * scores into a probability distribution ("how much of my attention does
     * each position get"). Same math as softmaxCrossEntropy's first half, but
     * differentiable on its own because here the output feeds further layers.
     */
    public function softmaxRows(): self
    {
        $data = [];
        foreach ($this->data as $rowIndex => $logits) {
            $highest = max($logits);
            $exponentials = [];
            $sum = 0.0;
            foreach ($logits as $columnIndex => $logit) {
                $e = exp($logit - $highest);
                $exponentials[$columnIndex] = $e;
                $sum += $e;
            }
            foreach ($exponentials as $columnIndex => $e) {
                $data[$rowIndex][$columnIndex] = $e / $sum;
            }
        }
        $result = new self($data);
        $result->parents = [$this];
        $result->backwardFunction = function () use ($result): void {
            // For each row: dscore_j = p_j × (upstream_j − Σ_k p_k × upstream_k)
            foreach ($result->gradient as $rowIndex => $upstreamRow) {
                $weightedSum = 0.0;
                foreach ($upstreamRow as $columnIndex => $upstream) {
                    $weightedSum += $result->data[$rowIndex][$columnIndex] * $upstream;
                }
                foreach ($upstreamRow as $columnIndex => $upstream) {
                    $this->gradient[$rowIndex][$columnIndex] +=
                        $result->data[$rowIndex][$columnIndex] * ($upstream - $weightedSum);
                }
            }
        };

        return $result;
    }

    /**
     * Glue tensors side by side (same row count) — how multi-head attention
     * reunites its heads: each contributes a slice of columns. Backward slices
     * the gradient back apart at the same offsets.
     *
     * @param array<int, Tensor> $tensors
     */
    public static function concatenateColumns(array $tensors): self
    {
        $data = [];
        foreach ($tensors[0]->data as $rowIndex => $unused) {
            $flat = [];
            foreach ($tensors as $tensor) {
                foreach ($tensor->data[$rowIndex] as $value) {
                    $flat[] = $value;
                }
            }
            $data[$rowIndex] = $flat;
        }
        $result = new self($data);
        $result->parents = $tensors;
        $result->backwardFunction = function () use ($tensors, $result): void {
            $offset = 0;
            foreach ($tensors as $tensor) {
                $width = $tensor->columnCount();
                foreach ($tensor->gradient as $rowIndex => $row) {
                    for ($column = 0; $column < $width; $column++) {
                        $tensor->gradient[$rowIndex][$column] += $result->gradient[$rowIndex][$offset + $column];
                    }
                }
                $offset += $width;
            }
        };

        return $result;
    }

    /**
     * Layer normalization — the stabilizer that makes DEEP networks trainable.
     *
     * Per row (per position): shift and scale the values to mean 0, variance 1,
     * then apply a learned per-column $gain and $bias (both 1×n) so the network
     * can undo the standardization wherever it helps. Deep stacks of layers
     * drift toward exploding or vanishing activations; renormalizing at every
     * block keeps every layer operating in its responsive range.
     *
     * The backward pass must account for each value's effect on its row's mean
     * and variance too — the formula below is the chain rule worked through
     * that dependency (and verified numerically in tests/check-ch8.php).
     */
    public function layerNormalizeRows(self $gain, self $bias): self
    {
        $epsilon = 1e-5;
        $columnCount = $this->columnCount();
        $data = [];
        $normalized = [];
        $inverseSpreads = [];
        foreach ($this->data as $rowIndex => $row) {
            $mean = array_sum($row) / $columnCount;
            $variance = 0.0;
            foreach ($row as $value) {
                $variance += ($value - $mean) ** 2;
            }
            $variance /= $columnCount;
            $inverseSpread = 1.0 / sqrt($variance + $epsilon);
            $inverseSpreads[$rowIndex] = $inverseSpread;
            foreach ($row as $columnIndex => $value) {
                $standardized = ($value - $mean) * $inverseSpread;
                $normalized[$rowIndex][$columnIndex] = $standardized;
                $data[$rowIndex][$columnIndex] = $standardized * $gain->data[0][$columnIndex] + $bias->data[0][$columnIndex];
            }
        }
        $result = new self($data);
        $result->parents = [$this, $gain, $bias];
        $result->backwardFunction = function () use ($gain, $bias, $normalized, $inverseSpreads, $columnCount, $result): void {
            foreach ($result->gradient as $rowIndex => $upstreamRow) {
                $scaledUpstream = [];
                $meanScaled = 0.0;
                $meanScaledTimesNormalized = 0.0;
                foreach ($upstreamRow as $columnIndex => $upstream) {
                    $scaled = $upstream * $gain->data[0][$columnIndex];
                    $scaledUpstream[$columnIndex] = $scaled;
                    $meanScaled += $scaled;
                    $meanScaledTimesNormalized += $scaled * $normalized[$rowIndex][$columnIndex];

                    $gain->gradient[0][$columnIndex] += $upstream * $normalized[$rowIndex][$columnIndex];
                    $bias->gradient[0][$columnIndex] += $upstream;
                }
                $meanScaled /= $columnCount;
                $meanScaledTimesNormalized /= $columnCount;
                foreach ($scaledUpstream as $columnIndex => $scaled) {
                    $this->gradient[$rowIndex][$columnIndex] += $inverseSpreads[$rowIndex]
                        * ($scaled - $meanScaled - $normalized[$rowIndex][$columnIndex] * $meanScaledTimesNormalized);
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
