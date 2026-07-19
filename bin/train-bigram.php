<?php

declare(strict_types=1);

require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/BigramModel.php';

use Llm\BigramModel;
use Llm\CharacterTokenizer;
use Llm\RandomNumberGenerator;

// Chapter 3: train the bigram model on 32k names and see what counting can do.

$text = file_get_contents(__DIR__ . '/../data/names.txt');
$names = explode("\n", trim($text));
$tokenizer = CharacterTokenizer::fromText($text);

$startedAt = hrtime(true);
$model = BigramModel::trainOn($names, $tokenizer);
printf("\"Trained\" on %s names in %.0f ms (training = counting)\n\n", number_format(count($names)), (hrtime(true) - $startedAt) / 1e6);

// ---- What did it learn? ----------------------------------------------------
$strongest = [];
foreach ($model->transitionCounts() as $currentTokenId => $row) {
    foreach ($row as $nextTokenId => $count) {
        $strongest[] = [$count, $currentTokenId, $nextTokenId];
    }
}
usort($strongest, fn ($a, $b) => $b[0] <=> $a[0]);

$label = fn (int $tokenId): string => $tokenizer->vocabulary()[$tokenId] === "\n" ? '␃' : $tokenizer->vocabulary()[$tokenId];
echo "Most common transitions (␃ = name boundary):\n";
foreach (array_slice($strongest, 0, 8) as [$count, $currentTokenId, $nextTokenId]) {
    printf("  %s → %s  %s times\n", $label($currentTokenId), $label($nextTokenId), number_format($count));
}

// ---- Generate --------------------------------------------------------------
$random = new RandomNumberGenerator(2026);
echo "\n20 generated names (seed 2026):\n";
for ($i = 0; $i < 20; $i++) {
    printf("  %s\n", $model->generate($random));
}

// ---- Measure ---------------------------------------------------------------
$uniformBaseline = log($tokenizer->vocabularySize());
$loss = $model->averageNegativeLogLikelihood($names);
printf("\nLoss (average negative log-likelihood per character):\n");
printf("  know-nothing baseline (uniform over %d tokens): %.4f\n", $tokenizer->vocabularySize(), $uniformBaseline);
printf("  trained bigram model:                           %.4f\n", $loss);
printf("Lower is better. This one number is what every model that follows will fight to reduce.\n");

// ---- Export for the textbook heatmap --------------------------------------
$export = [
    'vocabulary' => array_map(fn (string $c) => $c === "\n" ? '·' : $c, $tokenizer->vocabulary()),
    'probabilities' => array_map(
        fn (array $row) => array_map(fn (float $p) => round($p, 4), $row),
        $model->transitionProbabilities()
    ),
];
file_put_contents(__DIR__ . '/../docs/bigram-data.json', json_encode($export));
echo "\nWrote docs/bigram-data.json (probability table for the textbook's heatmap).\n";
