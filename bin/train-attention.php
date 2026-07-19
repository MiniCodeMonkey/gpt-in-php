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

// Chapter 7: train a single attention head on the names — not to win the loss
// ladder (one head with no feed-forward layer is just the engine, no chassis),
// but to watch REAL attention patterns form. The full transformer is chapter 8.

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

/** A name becomes ONE training sequence: inputs [·, e, m, m, a], targets [e, m, m, a, ·]. */
$buildSequence = function (string $name) use ($tokenizer, $contextLength): array {
    $tokenIds = [0, ...$tokenizer->encode($name), 0];
    $tokenIds = array_slice($tokenIds, 0, $contextLength + 1);

    return [array_slice($tokenIds, 0, -1), array_slice($tokenIds, 1)];
};

$model = new AttentionLanguageModel($tokenizer->vocabularySize(), $contextLength, 16, $random);
printf("Model: token+position embeddings (16-dim) → 1 attention head → logits (%s parameters)\n", number_format($model->parameterCount()));
printf("Training on %s names, validating on %s held-out names\n\n", number_format(count($trainingNames)), number_format(count($validationNames)));

$evaluate = function (array $names) use ($model, $buildSequence): float {
    $totalLoss = 0.0;
    $totalTargets = 0;
    foreach ($names as $name) {
        [$inputIds, $targetIds] = $buildSequence($name);
        $totalLoss += $model->computeLoss($inputIds, $targetIds)->data[0][0] * count($targetIds);
        $totalTargets += count($targetIds);
    }

    return $totalLoss / $totalTargets;
};
$validationSample = array_slice($validationNames, 0, 800); // full val eval is slow at 1 name/pass — sample it

$namesPerStep = 32;
$totalSteps = 4000;
$trainingNameCount = count($trainingNames);
$startedAt = hrtime(true);
for ($step = 1; $step <= $totalSteps; $step++) {
    $learningRate = ($step <= 3000 ? 0.4 : 0.08) / $namesPerStep;

    $model->zeroGradients();
    for ($i = 0; $i < $namesPerStep; $i++) {
        $name = $trainingNames[(int) floor($random->nextFloat() * $trainingNameCount)];
        [$inputIds, $targetIds] = $buildSequence($name);
        $model->computeLoss($inputIds, $targetIds)->backward(); // gradients ACCUMULATE across the 32 names
    }
    $model->applyGradientStep($learningRate);

    if ($step % 500 === 0 || $step === 1) {
        printf("  step %4d   validation loss %.4f\n", $step, $evaluate($validationSample));
    }
}
printf("\nTrained %d steps (%d names each) in %.0fs\n", $totalSteps, $namesPerStep, (hrtime(true) - $startedAt) / 1e9);
printf("Final validation loss (sampled): %.4f\n", $evaluate($validationSample));
printf("For reference: bigram 2.4546, MLP 2.1932. One head alone ≈ MLP-grade;\nthe full transformer (attention + feed-forward, stacked) comes next chapter.\n\n");

echo "15 generated names (seed 7):\n";
$generator = new RandomNumberGenerator(7);
for ($i = 0; $i < 15; $i++) {
    printf("  %s\n", $model->generate($generator, $tokenizer));
}

// ---- Export real attention patterns for the textbook -----------------------
$heatmaps = [];
foreach (['mathias', 'isabella', 'alexander', 'sofia'] as $name) {
    [$inputIds] = $buildSequence($name);
    $model->computeLogits($inputIds);
    $labels = array_map(
        fn (int $tokenId) => $tokenizer->vocabulary()[$tokenId] === "\n" ? '·' : $tokenizer->vocabulary()[$tokenId],
        $inputIds
    );
    $heatmaps[] = [
        'name' => $name,
        'labels' => $labels,
        'weights' => array_map(
            fn (array $row) => array_map(fn (float $weight) => round($weight, 4), $row),
            $model->attentionHead->lastAttentionWeights
        ),
    ];
}
file_put_contents(__DIR__ . '/../docs/attention-data.json', json_encode($heatmaps));
echo "\nWrote docs/attention-data.json (real attention patterns for the textbook).\n";
