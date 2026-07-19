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

// Chapter 6: train the MLP language model — the first attempt to break the
// bigram's 2.4546 ceiling by reading THREE characters of context.

$text = file_get_contents(__DIR__ . '/../data/names.txt');
$names = explode("\n", trim($text));
$tokenizer = CharacterTokenizer::fromText($text);

$contextLength = 3;

// ---- Train/validation split -------------------------------------------------
// New discipline, needed from now on: hold some names out of training entirely.
// Training loss measures what the model has SEEN; validation loss measures what
// it can GENERALIZE to. A model can drive training loss to zero by memorizing —
// validation loss is the score that can't be cheated.
// (Split by NAME, not by window — windows from one name overlap, and letting a
//  name straddle the split would leak training data into the exam.)
$random = new RandomNumberGenerator(2026);
$shuffled = $names;
for ($i = count($shuffled) - 1; $i > 0; $i--) { // seeded Fisher–Yates shuffle
    $j = (int) floor($random->nextFloat() * ($i + 1));
    [$shuffled[$i], $shuffled[$j]] = [$shuffled[$j], $shuffled[$i]];
}
$validationCount = (int) floor(count($shuffled) * 0.1);
$validationNames = array_slice($shuffled, 0, $validationCount);
$trainingNames = array_slice($shuffled, $validationCount);

/** Every (context window → next token) example a list of names contains. */
$buildExamples = function (array $names) use ($tokenizer, $contextLength): array {
    $contexts = [];
    $targets = [];
    foreach ($names as $name) {
        $window = array_fill(0, $contextLength, 0); // "emma" ⇒ [···]→e, [··e]→m, [·em]→m, [emm]→a, [mma]→·
        foreach ([...$tokenizer->encode($name), 0] as $targetId) {
            $contexts[] = $window;
            $targets[] = $targetId;
            $window = [...array_slice($window, 1), $targetId];
        }
    }

    return [$contexts, $targets];
};
[$trainingContexts, $trainingTargets] = $buildExamples($trainingNames);
[$validationContexts, $validationTargets] = $buildExamples($validationNames);
printf("Training on %s examples from %s names; validating on %s examples from %s held-out names\n",
    number_format(count($trainingContexts)), number_format(count($trainingNames)),
    number_format(count($validationContexts)), number_format(count($validationNames)));

$model = new MultiLayerPerceptronLanguageModel(
    vocabularySize: $tokenizer->vocabularySize(),
    contextLength: $contextLength,
    embeddingDimensions: 8,
    hiddenNeuronCount: 128,
    random: $random,
);
printf("Model: 3 chars → 8-dim embeddings → 128 hidden neurons → 27 logits (%s parameters)\n\n", number_format($model->parameterCount()));

$evaluate = function (MultiLayerPerceptronLanguageModel $modelToScore, array $contexts, array $targets): float {
    $chunkSize = 10000;
    $weightedSum = 0.0;
    for ($start = 0; $start < count($contexts); $start += $chunkSize) {
        $chunkContexts = array_slice($contexts, $start, $chunkSize);
        $chunkTargets = array_slice($targets, $start, $chunkSize);
        $weightedSum += $modelToScore->computeLoss($chunkContexts, $chunkTargets)->data[0][0] * count($chunkContexts);
    }

    return $weightedSum / count($contexts);
};

// ---- The training loop ------------------------------------------------------
$batchSize = 400;
$totalSteps = 6000;
$trainingExampleCount = count($trainingContexts);
$curve = [];
$startedAt = hrtime(true);
for ($step = 1; $step <= $totalSteps; $step++) {
    $learningRate = $step <= 4000 ? 0.5 : 0.05;

    $batchContexts = [];
    $batchTargets = [];
    for ($i = 0; $i < $batchSize; $i++) {
        $pick = (int) floor($random->nextFloat() * $trainingExampleCount);
        $batchContexts[] = $trainingContexts[$pick];
        $batchTargets[] = $trainingTargets[$pick];
    }

    $model->zeroGradients();
    $loss = $model->computeLoss($batchContexts, $batchTargets);
    $loss->backward();
    $model->applyGradientStep($learningRate);

    if ($step % 250 === 0 || $step === 1) {
        $validationLoss = $evaluate($model, $validationContexts, $validationTargets);
        $curve[] = ['step' => $step, 'batch' => round($loss->data[0][0], 4), 'validation' => round($validationLoss, 4)];
        printf("  step %4d   batch loss %.4f   validation loss %.4f\n", $step, $loss->data[0][0], $validationLoss);
    }
}
printf("\nTrained %d steps in %.0fs\n", $totalSteps, (hrtime(true) - $startedAt) / 1e9);

$trainingLoss = $evaluate($model, $trainingContexts, $trainingTargets);
$validationLoss = $evaluate($model, $validationContexts, $validationTargets);
printf("Final:  training loss %.4f   validation loss %.4f\n", $trainingLoss, $validationLoss);
printf("Ladder: know-nothing 3.2958 → bigram 2.4546 → MLP %.4f\n\n", $validationLoss);

echo "20 generated names (seed 7):\n";
$generator = new RandomNumberGenerator(7);
$samples = [];
for ($i = 0; $i < 20; $i++) {
    $samples[] = $model->generate($generator, $tokenizer);
    printf("  %s\n", $samples[$i]);
}

// ---- Bonus run for the textbook: 2-dimensional embeddings ------------------
// With embeddingDimensions = 2, every character's embedding is literally a
// point on a plane we can draw. Costs some loss (2 numbers can't hold much
// nuance) but shows what embedding-learning actually does.
echo "\nTraining a second model with 2-dim embeddings for the textbook's map…\n";
$vizModel = new MultiLayerPerceptronLanguageModel($tokenizer->vocabularySize(), $contextLength, 2, 64, $random);
for ($step = 1; $step <= 4000; $step++) {
    $learningRate = $step <= 3000 ? 0.5 : 0.05;
    $batchContexts = [];
    $batchTargets = [];
    for ($i = 0; $i < $batchSize; $i++) {
        $pick = (int) floor($random->nextFloat() * $trainingExampleCount);
        $batchContexts[] = $trainingContexts[$pick];
        $batchTargets[] = $trainingTargets[$pick];
    }
    $vizModel->zeroGradients();
    $loss = $vizModel->computeLoss($batchContexts, $batchTargets);
    $loss->backward();
    $vizModel->applyGradientStep($learningRate);
}
printf("2-dim model validation loss: %.4f (worse than 8-dim, as expected — the plane is cramped)\n",
    $evaluate($vizModel, $validationContexts, $validationTargets));

$embeddingPoints = [];
foreach ($tokenizer->vocabulary() as $tokenId => $character) {
    $embeddingPoints[] = [
        'character' => $character === "\n" ? '·' : $character,
        'x' => round($vizModel->embeddingTable->data[$tokenId][0], 4),
        'y' => round($vizModel->embeddingTable->data[$tokenId][1], 4),
    ];
}

file_put_contents(__DIR__ . '/../docs/mlp-training.json', json_encode([
    'curve' => $curve,
    'finalTrainingLoss' => round($trainingLoss, 4),
    'finalValidationLoss' => round($validationLoss, 4),
    'samples' => $samples,
    'embeddingPoints' => $embeddingPoints,
]));
echo "\nWrote docs/mlp-training.json (loss curve + embedding map for the textbook).\n";
