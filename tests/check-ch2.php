<?php

declare(strict_types=1);

require __DIR__ . '/../src/CharacterTokenizer.php';
require __DIR__ . '/../src/BytePairEncoder.php';

use Llm\BytePairEncoder;
use Llm\CharacterTokenizer;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

// ---- CharacterTokenizer ----------------------------------------------------
$tokenizer = CharacterTokenizer::fromText("hello\nworld");

check($tokenizer->vocabularySize() === 8, 'vocabulary has 8 distinct characters (\n d e h l o r w)');
check($tokenizer->vocabulary()[0] === "\n", 'IDs are assigned in sorted order, newline first');
check($tokenizer->decode($tokenizer->encode('hello')) === 'hello', 'decode(encode(x)) round-trips');
check($tokenizer->encode('hell') === array_reverse(array_reverse($tokenizer->encode('hell'))), 'encode is deterministic');

$threw = false;
try {
    $tokenizer->encode('hello!');
} catch (\InvalidArgumentException) {
    $threw = true;
}
check($threw, 'encoding an out-of-vocabulary character ("!") fails loudly, not silently');

// ---- BytePairEncoder -------------------------------------------------------
$bytePairEncoder = BytePairEncoder::train("hug hug hug hug pug pug hugs", 2);

// "hu"/"ug" appears in hug(4) + hugs(1) = 5 words vs pug(2): first merge must be the most frequent pair.
$firstMerge = $bytePairEncoder->merges()[0];
check(in_array($firstMerge, [['h', 'u'], ['u', 'g']], true), 'first merge is the most frequent adjacent pair');
check(count($bytePairEncoder->merges()) === 2, 'training learned exactly the requested number of merges');
check(implode('', $bytePairEncoder->tokenizeWord('hugpug')) === 'hugpug', 'gluing tokens back together restores the word');
check(count($bytePairEncoder->tokenizeWord('hug')) < 3, '"hug" now takes fewer tokens than its 3 characters');
check($bytePairEncoder->tokenizeWord('zzz') === ['z', 'z', 'z'], 'unseen characters still tokenize (as single characters)');

echo "\nAll checks passed. Explore with: php bin/explore-tokenizer.php\n";
