<?php

declare(strict_types=1);

require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/BytePairEncoder.php';

use Llm\BytePairEncoder;
use Llm\CharacterTokenizer;

// Chapter 2 exploration: tokenize the names dataset two ways.

$text = file_get_contents(__DIR__ . '/../data/names.txt');
$names = explode("\n", trim($text));
printf("Corpus: %s names, %s characters\n\n", number_format(count($names)), number_format(strlen($text)));

// ---- 1. Character-level tokenizer ----------------------------------------
$tokenizer = CharacterTokenizer::fromText($text);
printf("Character vocabulary (%d tokens):\n", $tokenizer->vocabularySize());
foreach ($tokenizer->vocabulary() as $tokenId => $character) {
    printf("  %2d → %s\n", $tokenId, $character === "\n" ? '\n  (newline — our "end of name" marker)' : $character);
}

$example = 'mathias';
$tokenIds = $tokenizer->encode($example);
printf("\nencode(\"%s\") → [%s]\n", $example, implode(', ', $tokenIds));
printf("decode(back)     → \"%s\"  (round trip %s)\n\n", $tokenizer->decode($tokenIds),
    $tokenizer->decode($tokenIds) === $example ? 'OK' : 'BROKEN');

// ---- 2. Byte-pair encoder -------------------------------------------------
$numberOfMerges = 40;
$startedAt = hrtime(true);
$bytePairEncoder = BytePairEncoder::train($text, $numberOfMerges);
printf("Trained BPE with %d merges in %.1fs. First 15 merges learned:\n",
    $numberOfMerges, (hrtime(true) - $startedAt) / 1e9);

foreach (array_slice($bytePairEncoder->merges(), 0, 15) as $index => [$first, $second]) {
    printf("  merge %2d: \"%s\" + \"%s\" → \"%s\"\n", $index + 1, $first, $second, $first . $second);
}

echo "\nNames tokenized with the learned merges:\n";
foreach (['isabella', 'sophia', 'mathias', 'christopher'] as $name) {
    printf("  %-12s → [%s]\n", $name, implode('][', $bytePairEncoder->tokenizeWord($name)));
}

// Compression: how many tokens per name, before vs after BPE?
$characterTokens = 0;
$bpeTokens = 0;
foreach ($names as $name) {
    $characterTokens += strlen($name);
    $bpeTokens += count($bytePairEncoder->tokenizeWord($name));
}
printf("\nAverage tokens per name: %.2f as characters → %.2f after %d merges (%.0f%% shorter)\n",
    $characterTokens / count($names), $bpeTokens / count($names), $numberOfMerges,
    100 * (1 - $bpeTokens / $characterTokens));
echo "Shorter sequences = less compute per name. That trade (bigger vocabulary\nfor shorter sequences) is exactly why frontier models use ~100k-token vocabularies.\n";
