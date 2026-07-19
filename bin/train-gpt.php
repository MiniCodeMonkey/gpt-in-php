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

// Chapter 8: the summit. Train the full GPT on names and take the ladder's
// last rung. Saves the trained model to models/gpt-names.json — chapters 9
// and 10 build on these exact weights.

$text = file_get_contents(__DIR__ . '/../data/names.txt');
$names = explode("\n", trim($text));
$tokenizer = CharacterTokenizer::fromText($text);
$contextLength = 12;

$random = new RandomNumberGenerator(2026);
$shuffled = $names;
for ($i = count($shuffled) - 1; $i > 0; $i--) {
    $j = (int) floor($random->nextFloat() * ($i + 1));
    [$shuffled[$i], $shuffled[$j]] = [$shuffled[$j], $shuffled[$i]];
}
$validationCount = (int) floor(count($shuffled) * 0.1);
$validationNames = array_slice($shuffled, 0, $validationCount);
$trainingNames = array_slice($shuffled, $validationCount);

$buildSequence = function (string $name) use ($tokenizer, $contextLength): array {
    $tokenIds = array_slice([0, ...$tokenizer->encode($name), 0], 0, $contextLength + 1);

    return [array_slice($tokenIds, 0, -1), array_slice($tokenIds, 1)];
};

$model = new GptLanguageModel(
    vocabularySize: $tokenizer->vocabularySize(),
    contextLength: $contextLength,
    embeddingDimensions: 32,
    headCount: 4,
    blockCount: 2,
    random: $random,
);
printf("GPT: 32-dim, 4 heads × 2 blocks, context %d — %s parameters\n", $contextLength, number_format($model->parameterCount()));
printf("(GPT-2: 768-dim, 12 heads × 12 blocks, context 1024 — 124M parameters. Same blueprint.)\n\n");

$evaluate = function (array $namesToScore) use ($model, $buildSequence): float {
    $totalLoss = 0.0;
    $totalTargets = 0;
    foreach ($namesToScore as $name) {
        [$inputIds, $targetIds] = $buildSequence($name);
        $totalLoss += $model->computeLoss($inputIds, $targetIds)->data[0][0] * count($targetIds);
        $totalTargets += count($targetIds);
    }

    return $totalLoss / $totalTargets;
};
$validationSample = array_slice($validationNames, 0, 600);

$namesPerStep = 32;
$totalSteps = 5000;
$trainingNameCount = count($trainingNames);
$curve = [];
$startedAt = hrtime(true);
for ($step = 1; $step <= $totalSteps; $step++) {
    $learningRate = ($step <= 4000 ? 0.3 : 0.06) / $namesPerStep;

    $model->zeroGradients();
    $batchLoss = 0.0;
    for ($i = 0; $i < $namesPerStep; $i++) {
        $name = $trainingNames[(int) floor($random->nextFloat() * $trainingNameCount)];
        [$inputIds, $targetIds] = $buildSequence($name);
        $loss = $model->computeLoss($inputIds, $targetIds);
        $loss->backward();
        $batchLoss += $loss->data[0][0] / $namesPerStep;
    }
    $model->applyGradientStep($learningRate);

    if ($step % 500 === 0 || $step === 1) {
        $validationLoss = $evaluate($validationSample);
        $curve[] = ['step' => $step, 'batch' => round($batchLoss, 4), 'validation' => round($validationLoss, 4)];
        printf("  step %4d   batch loss %.4f   validation loss %.4f   (%.0fs elapsed)\n",
            $step, $batchLoss, $validationLoss, (hrtime(true) - $startedAt) / 1e9);
    }
}
printf("\nTrained %d steps × %d names in %.0fs\n", $totalSteps, $namesPerStep, (hrtime(true) - $startedAt) / 1e9);

$finalValidationLoss = $evaluate($validationNames);
printf("Final validation loss (ALL %s held-out names): %.4f\n\n", number_format(count($validationNames)), $finalValidationLoss);
printf("The ladder:  know-nothing 3.2958 → bigram 2.4546 → MLP 2.1932 → GPT %.4f\n\n", $finalValidationLoss);

echo "25 generated names (seed 7):\n";
$generator = new RandomNumberGenerator(7);
$samples = [];
for ($i = 0; $i < 25; $i++) {
    $samples[] = $model->generate($generator, $tokenizer);
    printf("  %s\n", $samples[$i]);
}

@mkdir(__DIR__ . '/../models');
$model->saveToFile(__DIR__ . '/../models/gpt-names.json');
echo "\nSaved trained weights to models/gpt-names.json (chapters 9-10 reuse them).\n";

file_put_contents(__DIR__ . '/../docs/gpt-training.json', json_encode([
    'curve' => $curve,
    'finalValidationLoss' => round($finalValidationLoss, 4),
    'parameterCount' => $model->parameterCount(),
    'samples' => $samples,
]));
echo "Wrote docs/gpt-training.json for the textbook.\n";
