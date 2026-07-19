<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';

use Llm\CharacterTokenizer;
use Llm\RandomNumberGenerator;
use Llm\Tensor;

// Chapter 5 experiment: the bigram model AGAIN — but this time nothing is
// counted. A 27×27 weight matrix starts as random noise, and gradient descent
// must rediscover, from scratch, everything chapter 3 got by tallying.
//
// Same model class, two ways to fill in the numbers. If the loss converges to
// chapter 3's 2.4546, we'll know the learning machinery works end to end.

$text = file_get_contents(__DIR__ . '/../data/names.txt');
$names = explode("\n", trim($text));
$tokenizer = CharacterTokenizer::fromText($text);

// Every consecutive pair in the dataset: input character ID → target character ID.
$inputIds = [];
$targetIds = [];
foreach ($names as $name) {
    $tokenIds = [0, ...$tokenizer->encode($name), 0];
    for ($i = 0; $i < count($tokenIds) - 1; $i++) {
        $inputIds[] = $tokenIds[$i];
        $targetIds[] = $tokenIds[$i + 1];
    }
}
$pairCount = count($inputIds);
printf("Dataset: %s (input → target) pairs\n", number_format($pairCount));

// The model: one weight matrix. Row i holds the raw scores ("logits") that
// character i gives to every possible next character. Forward pass for a batch
// is selectRows (fetch each input's score row) + softmaxCrossEntropy. That is
// a bigram table again — but expressed as a differentiable function.
$random = new RandomNumberGenerator(2026);
$weights = Tensor::random($tokenizer->vocabularySize(), $tokenizer->vocabularySize(), $random, 0.01);

// Evaluate on all 228k pairs in chunks — one giant tensor (plus its equally
// giant gradient buffer) would eat hundreds of MB for no benefit.
$fullLoss = function () use ($weights, $inputIds, $targetIds, $pairCount): float {
    $chunkSize = 20000;
    $weightedSum = 0.0;
    for ($start = 0; $start < $pairCount; $start += $chunkSize) {
        $chunkInputIds = array_slice($inputIds, $start, $chunkSize);
        $chunkTargetIds = array_slice($targetIds, $start, $chunkSize);
        $chunkLoss = $weights->selectRows($chunkInputIds)->softmaxCrossEntropy($chunkTargetIds)->data[0][0];
        $weightedSum += $chunkLoss * count($chunkInputIds);
    }

    return $weightedSum / $pairCount;
};

printf("Loss before training (random weights): %.4f  (know-nothing baseline: ln 27 = %.4f)\n\n", $fullLoss(), log(27));

$batchSize = 2000;
$startedAt = hrtime(true);
$totalSteps = 1500;
for ($step = 1; $step <= $totalSteps; $step++) {
    // Learning-rate decay: big confident steps early, small careful ones near the
    // valley floor. Every serious training run does some version of this.
    $learningRate = $step <= 1000 ? 15.0 : 3.0;
    // A random minibatch — the full 228k pairs every step would be accurate but
    // slow; 2,000 give a noisy-but-unbiased gradient. (Frontier models do the
    // same: nobody feeds the whole internet per step.)
    $batchInputIds = [];
    $batchTargetIds = [];
    for ($i = 0; $i < $batchSize; $i++) {
        $pick = (int) floor($random->nextFloat() * $pairCount);
        $batchInputIds[] = $inputIds[$pick];
        $batchTargetIds[] = $targetIds[$pick];
    }

    $weights->zeroGradient();
    $loss = $weights->selectRows($batchInputIds)->softmaxCrossEntropy($batchTargetIds);
    $loss->backward();
    foreach ($weights->gradient as $rowIndex => $row) {
        foreach ($row as $columnIndex => $gradient) {
            $weights->data[$rowIndex][$columnIndex] -= $learningRate * $gradient;
        }
    }

    if ($step === 1 || $step % 250 === 0) {
        printf("  step %3d   batch loss %.4f\n", $step, $loss->data[0][0]);
    }
}
printf("\nTrained %d steps in %.1fs\n", $totalSteps, (hrtime(true) - $startedAt) / 1e9);
printf("Final loss on ALL %s pairs: %.4f   (chapter 3, by counting: 2.4546)\n", number_format($pairCount), $fullLoss());

// The learned weights ARE a bigram table: softmax each row and hand the
// probabilities to chapter 3's machinery for generation.
$probabilities = [];
foreach ($weights->data as $rowIndex => $logits) {
    $highest = max($logits);
    $exponentials = array_map(fn (float $l) => exp($l - $highest), $logits);
    $sum = array_sum($exponentials);
    $probabilities[$rowIndex] = array_map(fn (float $e) => $e / $sum, $exponentials);
}

echo "\n10 names generated from the LEARNED weights (seed 7):\n";
$generator = new RandomNumberGenerator(7);
for ($i = 0; $i < 10; $i++) {
    $currentTokenId = 0;
    $name = '';
    for ($length = 0; $length < 24; $length++) {
        $nextTokenId = $generator->sampleFromDistribution($probabilities[$currentTokenId]);
        if ($nextTokenId === 0) {
            break;
        }
        $name .= $tokenizer->vocabulary()[$nextTokenId];
        $currentTokenId = $nextTokenId;
    }
    printf("  %s\n", $name);
}

echo "\nSame quality as chapter 3 — as it must be: same information, same ceiling.\n";
echo "The difference: THIS table was learned by gradient descent, and unlike\n";
echo "counting, this recipe scales to any architecture we can take gradients of.\n";
