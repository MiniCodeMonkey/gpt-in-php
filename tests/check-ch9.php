<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';
require __DIR__ . '/../src/SelfAttentionHead.php';
require __DIR__ . '/../src/TransformerBlock.php';
require __DIR__ . '/../src/GptLanguageModel.php';
require __DIR__ . '/../src/TokenSampler.php';

use Llm\GptLanguageModel;
use Llm\RandomNumberGenerator;
use Llm\TokenSampler;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

$logits = [2.0, 1.0, 0.5, -1.0, 3.0];

// ---- Temperature ------------------------------------------------------------
$greedy = new TokenSampler(temperature: 0.0);
$alwaysSame = true;
for ($i = 0; $i < 50; $i++) {
    $alwaysSame = $alwaysSame && $greedy->sampleFromLogits($logits, new RandomNumberGenerator($i)) === 4;
}
check($alwaysSame, 'temperature 0 is greedy: always the argmax, regardless of seed');

$countHits = function (float $temperature, ?int $topK = null) use ($logits): array {
    $sampler = new TokenSampler($temperature, $topK);
    $random = new RandomNumberGenerator(42);
    $counts = array_fill(0, count($logits), 0);
    for ($i = 0; $i < 20000; $i++) {
        $counts[$sampler->sampleFromLogits($logits, $random)]++;
    }

    return $counts;
};

$cold = $countHits(0.4);
$honest = $countHits(1.0);
$hot = $countHits(2.5);
check($cold[4] > $honest[4] && $honest[4] > $hot[4], 'lower temperature concentrates on the favorite; higher spreads out');
check($hot[3] > $honest[3] * 2, 'high temperature resurrects unlikely tokens');

$entropy = function (array $counts): float {
    $total = array_sum($counts);
    $entropy = 0.0;
    foreach ($counts as $count) {
        if ($count > 0) {
            $entropy -= ($count / $total) * log($count / $total);
        }
    }

    return $entropy;
};
check($entropy($cold) < $entropy($honest) && $entropy($honest) < $entropy($hot), 'entropy (measured surprise) rises monotonically with temperature');

// ---- Top-k ------------------------------------------------------------------
$topTwo = $countHits(1.0, topK: 2);
check($topTwo[1] === 0 && $topTwo[2] === 0 && $topTwo[3] === 0, 'top-k = 2 makes every token outside the top 2 impossible');
check($topTwo[4] > 0 && $topTwo[0] > 0, '…while both survivors still get sampled (renormalized, not argmaxed)');

// ---- On the real trained model ----------------------------------------------
$model = GptLanguageModel::loadFromFile(__DIR__ . '/../models/gpt-names.json');
$logitsReal = $model->nextTokenLogits([0, 13, 1]); // "·ma"
check(count($logitsReal) === 27, 'the saved chapter-8 model loads and produces 27 logits');

$sampler = new TokenSampler(0.8);
$a = $sampler->sampleFromLogits($logitsReal, new RandomNumberGenerator(3));
$b = $sampler->sampleFromLogits($logitsReal, new RandomNumberGenerator(3));
check($a === $b, 'sampling is deterministic given the same seed');

echo "\nAll checks passed. Play: php bin/generate.php --temperature=1.5 --count=10\n";
