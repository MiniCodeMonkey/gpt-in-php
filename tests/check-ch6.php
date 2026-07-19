<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';
require __DIR__ . '/../src/MultiLayerPerceptronLanguageModel.php';

use Llm\CharacterTokenizer;
use Llm\MultiLayerPerceptronLanguageModel;
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

// ---- selectAndConcatenateRows ----------------------------------------------
$table = new Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
$windows = $table->selectAndConcatenateRows([[0, 2], [2, 2]]);
check($windows->data === [[1.0, 2.0, 5.0, 6.0], [5.0, 6.0, 5.0, 6.0]], 'windows concatenate the right embedding rows side by side');

$loss = $windows->softmaxCrossEntropy([1, 3]);
$loss->backward();
check(abs($table->gradient[1][0]) < 1e-15, 'unused embedding rows receive zero gradient');
check(abs($table->gradient[2][0]) > 0, 'a row used three times accumulates gradient from all three uses');

// Numerical verification through the full MLP: nudge one embedding entry.
$tokenizer = CharacterTokenizer::fromText("abc\n");
$buildAndMeasure = function (float $nudge) use ($tokenizer): array {
    $random = new RandomNumberGenerator(7);
    $model = new MultiLayerPerceptronLanguageModel($tokenizer->vocabularySize(), 3, 4, 8, $random);
    $model->embeddingTable->data[2][1] += $nudge;
    $loss = $model->computeLoss([[0, 2, 1], [2, 1, 3]], [3, 0]);

    return [$model, $loss];
};
[$model, $loss] = $buildAndMeasure(0.0);
$loss->backward();
$nudge = 1e-6;
$numerical = ($buildAndMeasure($nudge)[1]->data[0][0] - $loss->data[0][0]) / $nudge;
check(abs($numerical - $model->embeddingTable->gradient[2][1]) < 1e-5,
    sprintf('gradient flows correctly through embed→hidden→tanh→output (%.6f vs %.6f)', $model->embeddingTable->gradient[2][1], $numerical));

// ---- The model trains -------------------------------------------------------
$random = new RandomNumberGenerator(42);
$model = new MultiLayerPerceptronLanguageModel($tokenizer->vocabularySize(), 3, 4, 16, $random);
check($model->parameterCount() === 4 * 4 + 12 * 16 + 16 + 16 * 4 + 4, 'parameter count adds up (embeddings + weights + biases)');

$contexts = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]];
$targets = [1, 2, 3, 0]; // teach it the toy sequence \n a b c \n
$lossBefore = $model->computeLoss($contexts, $targets)->data[0][0];
for ($step = 0; $step < 200; $step++) {
    $model->zeroGradients();
    $loss = $model->computeLoss($contexts, $targets);
    $loss->backward();
    $model->applyGradientStep(0.5);
}
$lossAfter = $model->computeLoss($contexts, $targets)->data[0][0];
check($lossAfter < 0.1 && $lossBefore > 1.0, sprintf('200 steps memorize a toy sequence (loss %.3f → %.4f)', $lossBefore, $lossAfter));

$generated = $model->generate(new RandomNumberGenerator(3), $tokenizer);
check($generated === $model->generate(new RandomNumberGenerator(3), $tokenizer), 'generation is deterministic given a seed');
check($generated === 'abc', sprintf('after memorizing, it generates the sequence it learned ("%s")', $generated));

echo "\nAll checks passed. Real training: ./bin/phpj bin/train-mlp.php\n";
