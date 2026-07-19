<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';
require __DIR__ . '/../src/SelfAttentionHead.php';
require __DIR__ . '/../src/TransformerBlock.php';
require __DIR__ . '/../src/GptLanguageModel.php';

use Llm\CharacterTokenizer;
use Llm\GptLanguageModel;
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

// ---- concatenateColumns -----------------------------------------------------
$left = new Tensor([[1.0, 2.0], [3.0, 4.0]]);
$right = new Tensor([[5.0], [6.0]]);
$joined = Tensor::concatenateColumns([$left, $right]);
check($joined->data === [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], 'concatenateColumns glues tensors side by side');
$joined->softmaxCrossEntropy([2, 0])->backward();
check(abs($right->gradient[0][0]) > 0 && abs($left->gradient[0][0]) > 0, 'gradient slices back apart onto both sources');

// ---- layerNormalizeRows, numerically verified -------------------------------
$makeNormLoss = function (array $xData, array $gainData) {
    $x = new Tensor($xData);
    $gain = new Tensor($gainData);
    $bias = new Tensor([[0.1, -0.2, 0.3, 0.0]]);
    $loss = $x->layerNormalizeRows($gain, $bias)->tanh()->softmaxCrossEntropy([1, 3]);

    return [$loss, $x, $gain];
};
$xData = [[0.5, -1.0, 2.0, 0.3], [1.5, 0.2, -0.8, -1.1]];
$gainData = [[1.2, 0.9, 1.1, 0.8]];
[$loss, $x, $gain] = $makeNormLoss($xData, $gainData);
$loss->backward();

$nudge = 1e-6;
$maximumError = 0.0;
foreach ([[0, 0], [0, 2], [1, 3]] as [$row, $column]) {
    $nudged = $xData;
    $nudged[$row][$column] += $nudge;
    $numerical = ($makeNormLoss($nudged, $gainData)[0]->data[0][0] - $loss->data[0][0]) / $nudge;
    $maximumError = max($maximumError, abs($numerical - $x->gradient[$row][$column]));
}
$nudgedGain = $gainData;
$nudgedGain[0][1] += $nudge;
$numericalGain = ($makeNormLoss($xData, $nudgedGain)[0]->data[0][0] - $loss->data[0][0]) / $nudge;
$maximumError = max($maximumError, abs($numericalGain - $gain->gradient[0][1]));
check($maximumError < 1e-5, sprintf('layer norm gradients (through mean AND variance) match numerical nudging (max error %.1e)', $maximumError));

// ---- The full GPT, numerically verified -------------------------------------
$tokenizer = CharacterTokenizer::fromText("abcde\n");
$makeGpt = fn (): GptLanguageModel => new GptLanguageModel($tokenizer->vocabularySize(), 8, 8, 2, 2, new RandomNumberGenerator(13));
$inputIds = [0, 3, 1, 4, 2];
$targetIds = [3, 1, 4, 2, 0];

$model = $makeGpt();
$loss = $model->computeLoss($inputIds, $targetIds);
$loss->backward();

$nudged = $makeGpt();
$nudged->tokenEmbeddings->data[3][2] += $nudge;
$numerical = ($nudged->computeLoss($inputIds, $targetIds)->data[0][0] - $loss->data[0][0]) / $nudge;
check(abs($numerical - $model->tokenEmbeddings->gradient[3][2]) < 1e-4,
    sprintf('gradient flows through the ENTIRE 2-block transformer stack (%.6f vs %.6f)', $model->tokenEmbeddings->gradient[3][2], $numerical));

// ---- Causality survives the full stack --------------------------------------
$logitsBefore = $model->computeLogits([0, 3, 1, 4, 2])->data;
$logitsAfter = $model->computeLogits([0, 3, 1, 4, 5])->data;
$earlierUnchanged = true;
for ($position = 0; $position < 4; $position++) {
    foreach ($logitsBefore[$position] as $columnIndex => $value) {
        if (abs($value - $logitsAfter[$position][$columnIndex]) > 1e-10) {
            $earlierUnchanged = false;
        }
    }
}
check($earlierUnchanged, 'causality survives residuals, layer norm, multi-head, and stacking');

// ---- It learns, and persistence works ---------------------------------------
$random = new RandomNumberGenerator(3);
$model = $makeGpt();
$sequences = [];
foreach ([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 2, 4, 4, 1]] as $body) {
    $sequences[] = [[0, ...array_slice($body, 0, 4)], [...array_slice($body, 0, 4), 0]];
    $sequences[count($sequences) - 1][1] = [...$body];
}
$lossBefore = null;
$lossAfter = null;
for ($step = 0; $step < 300; $step++) {
    $model->zeroGradients();
    $stepLoss = 0.0;
    foreach ($sequences as [$sequenceInputs, $sequenceTargets]) {
        $loss = $model->computeLoss($sequenceInputs, $sequenceTargets);
        $loss->backward();
        $stepLoss += $loss->data[0][0] / count($sequences);
    }
    $model->applyGradientStep(0.1 / count($sequences));
    $lossBefore ??= $stepLoss;
    $lossAfter = $stepLoss;
}
check($lossAfter < 0.35 && $lossBefore > 1.5, sprintf('the full GPT memorizes toy sequences (loss %.2f → %.3f)', $lossBefore, $lossAfter));

$savePath = sys_get_temp_dir() . '/gpt-check-ch8.json';
$model->saveToFile($savePath);
$restored = GptLanguageModel::loadFromFile($savePath);
unlink($savePath);
$restoredLogits = $restored->computeLogits($inputIds)->data;
$originalLogits = $model->computeLogits($inputIds)->data;
$identical = true;
foreach ($originalLogits as $rowIndex => $row) {
    foreach ($row as $columnIndex => $value) {
        if (abs($value - $restoredLogits[$rowIndex][$columnIndex]) > 1e-12) {
            $identical = false;
        }
    }
}
check($identical, 'save → load → identical predictions (persistence round-trip)');

check(
    $model->generate(new RandomNumberGenerator(7), $tokenizer) === $restored->generate(new RandomNumberGenerator(7), $tokenizer),
    'the restored model generates exactly what the original does'
);

echo "\nAll checks passed. The big one: ./bin/phpj bin/train-gpt.php\n";
