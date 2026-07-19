<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';

use Llm\RandomNumberGenerator;
use Llm\Tensor;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

// ---- The star check, tensor edition ----------------------------------------
// Build a real mini network: loss = softmaxCrossEntropy(tanh(X·W + b), targets).
// Compute all gradients with backward(), then verify a sample of them by
// physically nudging individual entries and re-running the whole forward pass.

$xData = [[0.5, -1.2, 0.8], [1.5, 0.3, -0.7]];
$wData = [[0.1, -0.4, 0.6, 0.2], [0.7, 0.3, -0.2, -0.5], [-0.3, 0.8, 0.1, 0.4]];
$bData = [[0.05, -0.1, 0.2, 0.0]];
$targets = [2, 0];

$forwardLoss = function (array $x, array $w, array $b) use ($targets): float {
    $loss = (new Tensor($x))->multiply(new Tensor($w))->addRowVector(new Tensor($b))->tanh()->softmaxCrossEntropy($targets);

    return $loss->data[0][0];
};

$x = new Tensor($xData);
$w = new Tensor($wData);
$b = new Tensor($bData);
$loss = $x->multiply($w)->addRowVector($b)->tanh()->softmaxCrossEntropy($targets);
$loss->backward();

$nudge = 1e-6;
$maximumError = 0.0;
foreach ([[0, 1], [1, 2], [2, 3]] as [$row, $column]) { // sample entries of W
    $nudged = $wData;
    $nudged[$row][$column] += $nudge;
    $numerical = ($forwardLoss($xData, $nudged, $bData) - $loss->data[0][0]) / $nudge;
    $maximumError = max($maximumError, abs($numerical - $w->gradient[$row][$column]));
}
check($maximumError < 1e-5, sprintf('matmul→bias→tanh→softmaxCE gradients match numerical nudging (max error %.1e)', $maximumError));

$nudgedB = $bData;
$nudgedB[0][2] += $nudge;
$numericalBias = ($forwardLoss($xData, $wData, $nudgedB) - $loss->data[0][0]) / $nudge;
check(abs($numericalBias - $b->gradient[0][2]) < 1e-5, 'broadcast bias gradient collects correctly from every batch row');

// ---- softmax sanity ---------------------------------------------------------
$logits = new Tensor([[1.0, 2.0, 3.0]]);
$lossValue = $logits->softmaxCrossEntropy([2]);
$lossValue->backward();
$rowSum = array_sum($logits->gradient[0]);
check(abs($rowSum) < 1e-12, 'softmax gradient rows sum to zero (probability mass is only moved around)');
check($logits->gradient[0][2] < 0, "the correct token's logit is pushed UP (negative gradient = increase to lower loss)");
check($logits->gradient[0][0] > 0 && $logits->gradient[0][1] > 0, 'every wrong logit is pushed down');

// Equal logits → uniform probabilities → loss = ln(3), tying back to chapter 3's baseline.
$uniform = (new Tensor([[0.0, 0.0, 0.0]]))->softmaxCrossEntropy([1]);
check(abs($uniform->data[0][0] - log(3)) < 1e-12, 'equal scores give uniform probabilities: loss = ln(3), chapter 3\'s know-nothing baseline');

// ---- selectRows (the embedding lookup) --------------------------------------
$table = new Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
$selected = $table->selectRows([2, 0, 2]);
check($selected->data === [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]], 'selectRows fetches the right rows (an embedding lookup)');

$lossLike = $selected->softmaxCrossEntropy([0, 1, 1]);
$lossLike->backward();
$rowUsedTwiceGetsMore = abs($table->gradient[2][0]) > 0 && abs($table->gradient[1][0]) < 1e-15;
check($rowUsedTwiceGetsMore, 'gradients scatter back only onto rows that were actually used');

// ---- One real gradient descent step actually helps --------------------------
$random = new RandomNumberGenerator(99);
$weights = Tensor::random(5, 5, $random, 0.5);
$inputIds = [0, 1, 2, 3, 4, 0, 1];
$targetIds = [1, 2, 3, 4, 0, 1, 2];
$lossBefore = $weights->selectRows($inputIds)->softmaxCrossEntropy($targetIds)->data[0][0];
for ($step = 0; $step < 30; $step++) {
    $weights->zeroGradient();
    $loss = $weights->selectRows($inputIds)->softmaxCrossEntropy($targetIds);
    $loss->backward();
    foreach ($weights->gradient as $rowIndex => $row) {
        foreach ($row as $columnIndex => $gradient) {
            $weights->data[$rowIndex][$columnIndex] -= 5.0 * $gradient;
        }
    }
}
$lossAfter = $weights->selectRows($inputIds)->softmaxCrossEntropy($targetIds)->data[0][0];
check($lossAfter < $lossBefore / 2, sprintf('30 gradient steps on a toy problem: loss %.3f → %.3f', $lossBefore, $lossAfter));

echo "\nAll checks passed. Train the neural bigram: php bin/train-neural-bigram.php\n";
