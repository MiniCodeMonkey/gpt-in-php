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

use Llm\CharacterTokenizer;
use Llm\GptLanguageModel;
use Llm\RandomNumberGenerator;
use Llm\TokenSampler;

// Chapter 9: one frozen model, many personalities. Usage:
//   php bin/generate.php                            (the full gallery + export)
//   php bin/generate.php --temperature=1.3 --top-k=5 --count=10 --prefix=ma

$tokenizer = CharacterTokenizer::fromText(file_get_contents(__DIR__ . '/../data/names.txt'));
$model = GptLanguageModel::loadFromFile(__DIR__ . '/../models/gpt-names.json');

$generateName = function (TokenSampler $sampler, RandomNumberGenerator $random, string $prefix = '') use ($model, $tokenizer): string {
    $tokenIds = [0, ...$tokenizer->encode($prefix)];
    $name = $prefix;
    for ($length = strlen($prefix); $length < 24; $length++) {
        $window = array_slice($tokenIds, -$model->contextLength);
        $nextTokenId = $sampler->sampleFromLogits($model->nextTokenLogits($window), $random);
        if ($nextTokenId === 0) {
            break;
        }
        $name .= $tokenizer->vocabulary()[$nextTokenId];
        $tokenIds[] = $nextTokenId;
    }

    return $name;
};

$options = getopt('', ['temperature::', 'top-k::', 'count::', 'seed::', 'prefix::']);
if ($options !== []) {
    $sampler = new TokenSampler(
        temperature: (float) ($options['temperature'] ?? 1.0),
        keepTopK: isset($options['top-k']) ? (int) $options['top-k'] : null,
    );
    $random = new RandomNumberGenerator((int) ($options['seed'] ?? 7));
    for ($i = 0; $i < (int) ($options['count'] ?? 10); $i++) {
        printf("  %s\n", $generateName($sampler, $random, $options['prefix'] ?? ''));
    }
    exit(0);
}

// ---- The gallery ------------------------------------------------------------
$gallery = [];
foreach ([0.0, 0.5, 0.8, 1.0, 1.5] as $temperature) {
    $random = new RandomNumberGenerator(7);
    $sampler = new TokenSampler($temperature);
    $names = [];
    for ($i = 0; $i < 8; $i++) {
        $names[] = $generateName($sampler, $random);
    }
    $gallery[] = ['temperature' => $temperature, 'names' => $names];
    printf("temperature %.1f%s → %s\n", $temperature, $temperature === 0.0 ? ' (greedy)' : '', implode(', ', array_unique($names)));
}

echo "\nName completion (prefix conditioning — the seed of 'prompting'):\n";
$completions = [];
foreach (['ma', 'kai', 'zo'] as $prefix) {
    $random = new RandomNumberGenerator(11);
    $sampler = new TokenSampler(0.8);
    $names = [];
    for ($i = 0; $i < 6; $i++) {
        $names[] = $generateName($sampler, $random, $prefix);
    }
    $completions[] = ['prefix' => $prefix, 'names' => $names];
    printf("  %s… → %s\n", $prefix, implode(', ', $names));
}

// ---- Export real logits for the textbook's live temperature demo ------------
$context = 'mari';
$logits = $model->nextTokenLogits([0, ...$tokenizer->encode($context)]);
file_put_contents(__DIR__ . '/../docs/inference-data.json', json_encode([
    'context' => $context,
    'vocabulary' => array_map(fn (string $c) => $c === "\n" ? '·' : $c, $tokenizer->vocabulary()),
    'logits' => array_map(fn (float $l) => round($l, 4), $logits),
    'gallery' => $gallery,
    'completions' => $completions,
]));
echo "\nWrote docs/inference-data.json (real logits after \"{$context}\" for the textbook).\n";
