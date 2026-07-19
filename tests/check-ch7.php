<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';
require __DIR__ . '/../src/SelfAttentionHead.php';
require __DIR__ . '/../src/AttentionLanguageModel.php';

use Llm\AttentionLanguageModel;
use Llm\CharacterTokenizer;
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

$tokenizer = CharacterTokenizer::fromText("abcde\n");

// ---- New tensor operations, verified numerically through a full pipeline ----
$makeModel = function () use ($tokenizer): AttentionLanguageModel {
    return new AttentionLanguageModel($tokenizer->vocabularySize(), 8, 6, new RandomNumberGenerator(11));
};
$inputIds = [0, 3, 1, 2];
$targetIds = [3, 1, 2, 0];

$model = $makeModel();
$loss = $model->computeLoss($inputIds, $targetIds);
$loss->backward();

$nudge = 1e-6;
$maximumError = 0.0;
foreach ([['tokenEmbeddings', 3, 2], ['positionEmbeddings', 1, 4]] as [$tensorName, $row, $column]) {
    $nudged = $makeModel();
    $nudged->{$tensorName}->data[$row][$column] += $nudge;
    $numerical = ($nudged->computeLoss($inputIds, $targetIds)->data[0][0] - $loss->data[0][0]) / $nudge;
    $maximumError = max($maximumError, abs($numerical - $model->{$tensorName}->gradient[$row][$column]));
}
foreach ([['queryWeights', 2, 3], ['keyWeights', 0, 5], ['valueWeights', 4, 1]] as [$tensorName, $row, $column]) {
    $nudged = $makeModel();
    $nudged->attentionHead->{$tensorName}->data[$row][$column] += $nudge;
    $numerical = ($nudged->computeLoss($inputIds, $targetIds)->data[0][0] - $loss->data[0][0]) / $nudge;
    $maximumError = max($maximumError, abs($numerical - $model->attentionHead->{$tensorName}->gradient[$row][$column]));
}
check($maximumError < 1e-4, sprintf('gradients through the FULL attention pipeline match numerical nudging (max error %.1e)', $maximumError));

// ---- Attention weights are a proper causal pattern --------------------------
$weights = $model->attentionHead->lastAttentionWeights;
$isLowerTriangular = true;
$rowsSumToOne = true;
foreach ($weights as $rowIndex => $row) {
    $rowSum = 0.0;
    foreach ($row as $columnIndex => $weight) {
        if ($columnIndex > $rowIndex && abs($weight) > 1e-9) {
            $isLowerTriangular = false;
        }
        $rowSum += $weight;
    }
    if (abs($rowSum - 1.0) > 1e-9) {
        $rowsSumToOne = false;
    }
}
check($isLowerTriangular, 'no position attends to the future (mask works: upper triangle is ~0)');
check($rowsSumToOne, "each position's attention weights sum to 1 (it's a distribution over the visible past)");
check(abs($weights[0][0] - 1.0) < 1e-9, 'position 0 can only attend to itself, and does — weight exactly 1');

// ---- The causality property, tested end to end ------------------------------
// Changing a LATER token must not change any EARLIER position's predictions.
// (This is the property that lets one sequence act as T honest training examples.)
$logitsBefore = $model->computeLogits([0, 3, 1, 2])->data;
$logitsAfter = $model->computeLogits([0, 3, 1, 4])->data; // only position 3 differs
$earlierUnchanged = true;
for ($position = 0; $position < 3; $position++) {
    foreach ($logitsBefore[$position] as $columnIndex => $value) {
        if (abs($value - $logitsAfter[$position][$columnIndex]) > 1e-12) {
            $earlierUnchanged = false;
        }
    }
}
check($earlierUnchanged, 'changing token 4 leaves predictions at positions 1-3 EXACTLY unchanged (causality, end to end)');

$lastChanged = false;
foreach ($logitsBefore[3] as $columnIndex => $value) {
    if (abs($value - $logitsAfter[3][$columnIndex]) > 1e-9) {
        $lastChanged = true;
    }
}
check($lastChanged, "…while the changed position's own prediction does change");

// ---- It learns --------------------------------------------------------------
// A pattern a bigram/MLP-3 provably cannot learn: "the token after d repeats
// the FIRST token of the sequence". Requires looking back 4+ positions.
$random = new RandomNumberGenerator(5);
$trainingSequences = [];
foreach ([1, 2, 3, 4] as $firstTokenId) {
    $trainingSequences[] = [[0, $firstTokenId, 5, 5, 5, 4], [$firstTokenId, 5, 5, 5, 4, $firstTokenId]];
}
// (Note: position 0 sees only the boundary token, so it can never know which
//  first letter is coming — that alone puts an irreducible ln(4)/6 ≈ 0.23 floor
//  under the AVERAGE loss. So we test the recall position directly instead.
//  Also: 6 embedding dimensions proved too cramped to represent the lookup —
//  12 learn it easily. Capacity matters; we found this out empirically.)
$model = new AttentionLanguageModel($tokenizer->vocabularySize(), 8, 12, new RandomNumberGenerator(11));
for ($step = 0; $step < 1500; $step++) {
    $model->zeroGradients();
    foreach ($trainingSequences as [$sequenceInputs, $sequenceTargets]) {
        $model->computeLoss($sequenceInputs, $sequenceTargets)->backward();
    }
    $model->applyGradientStep(0.5 / count($trainingSequences));
}

$recallWorks = true;
$weakestRecall = 1.0;
foreach ($trainingSequences as [$sequenceInputs, $sequenceTargets]) {
    $logits = $model->computeLogits($sequenceInputs)->data[5]; // the position that must recall token #1
    $highest = max($logits);
    $exponentials = array_map(fn (float $logit) => exp($logit - $highest), $logits);
    $probability = $exponentials[$sequenceTargets[5]] / array_sum($exponentials);
    $weakestRecall = min($weakestRecall, $probability);
    $recallWorks = $recallWorks && $probability > 0.8;
}
check($recallWorks, sprintf('attention learns "recall the FIRST token from 5 positions back" — impossible for a 3-char window (weakest recall: %.0f%% confident)', $weakestRecall * 100));

$generated = $model->generate(new RandomNumberGenerator(9), $tokenizer);
check($generated === $model->generate(new RandomNumberGenerator(9), $tokenizer), 'generation is deterministic given a seed');

echo "\nAll checks passed. Real training: ./bin/phpj bin/train-attention.php\n";
