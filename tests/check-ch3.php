<?php

declare(strict_types=1);

require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/BigramModel.php';

use Llm\BigramModel;
use Llm\CharacterTokenizer;
use Llm\RandomNumberGenerator;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

// ---- RandomNumberGenerator -------------------------------------------------
$randomA = new RandomNumberGenerator(42);
$randomB = new RandomNumberGenerator(42);
check($randomA->nextFloat() === $randomB->nextFloat(), 'same seed produces the same sequence');

$random = new RandomNumberGenerator(42);
$inRange = true;
$drawCounts = [0, 0];
for ($i = 0; $i < 10000; $i++) {
    $value = $random->nextFloat();
    $inRange = $inRange && $value >= 0.0 && $value < 1.0;
    $drawCounts[$random->sampleFromDistribution([0.9, 0.1])]++;
}
check($inRange, 'nextFloat always lands in [0, 1)');
check($drawCounts[0] > 8700 && $drawCounts[0] < 9300, 'sampling respects probabilities (~90% of 10k draws hit the 0.9 option)');

// ---- BigramModel on a corpus tiny enough to verify by hand ------------------
$names = ['ab', 'ab', 'ac'];
$tokenizer = CharacterTokenizer::fromText(implode("\n", $names) . "\n");
$model = BigramModel::trainOn($names, $tokenizer, smoothing: 0.0);

// Vocabulary: \n=0 a=1 b=2 c=3. Transitions: a→b twice, a→c once.
check($model->transitionCounts()[1][2] === 2 && $model->transitionCounts()[1][3] === 1, 'counts tally transitions correctly (a→b ×2, a→c ×1)');
check(abs($model->transitionProbabilities()[1][2] - 2 / 3) < 1e-12, 'probabilities are counts divided by row totals (P(b|a) = 2/3)');

$rowSumsToOne = true;
foreach ($model->transitionProbabilities() as $row) {
    $rowTotal = array_sum($row);
    if ($rowTotal > 0 && abs($rowTotal - 1.0) > 1e-9) {
        $rowSumsToOne = false;
    }
}
check($rowSumsToOne, 'every probability row sums to 1 (it is a distribution)');

$smoothed = BigramModel::trainOn($names, $tokenizer, smoothing: 1.0);
check($smoothed->transitionProbabilities()[2][3] > 0.0, 'smoothing gives never-seen pairs (b→c) a small non-zero probability');

check(
    $smoothed->generate(new RandomNumberGenerator(7)) === $smoothed->generate(new RandomNumberGenerator(7)),
    'generation is deterministic given the same seed'
);

// ---- The loss --------------------------------------------------------------
// A uniform model (infinite smoothing limit) knows nothing: loss must be ln(vocabularySize).
$uniform = BigramModel::trainOn($names, $tokenizer, smoothing: 1e12);
check(abs($uniform->averageNegativeLogLikelihood($names) - log(4)) < 1e-6, 'a know-nothing (uniform) model scores exactly ln(vocabulary size)');
check($smoothed->averageNegativeLogLikelihood($names) < $uniform->averageNegativeLogLikelihood($names), 'training (counting) beats knowing nothing — lower loss');

echo "\nAll checks passed. Train on real names: php bin/train-bigram.php\n";
