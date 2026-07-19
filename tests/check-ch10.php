<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';
require __DIR__ . '/../src/WordTokenizer.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/Tensor.php';
require __DIR__ . '/../src/SelfAttentionHead.php';
require __DIR__ . '/../src/TransformerBlock.php';
require __DIR__ . '/../src/GptLanguageModel.php';
require __DIR__ . '/../src/TokenSampler.php';

use Llm\GptLanguageModel;
use Llm\RandomNumberGenerator;
use Llm\TokenSampler;
use Llm\WordTokenizer;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

// ---- WordTokenizer ----------------------------------------------------------
$tokenizer = WordTokenizer::fromText("<user> where is milo <bot> milo is here <end>");
check($tokenizer->vocabularySize() === 7, 'word tokenizer builds vocabulary from distinct words (7 here)');
check($tokenizer->decode($tokenizer->encode('milo is here')) === 'milo is here', 'word-level encode/decode round-trips');
check($tokenizer->knows('<bot>') && !$tokenizer->knows('elephant'), 'special tokens are ordinary vocabulary words; unknown words are detectable');

// ---- Toy world data ---------------------------------------------------------
$statementsPath = __DIR__ . '/../data/toy-world-statements.txt';
$chatPath = __DIR__ . '/../data/toy-world-chat.txt';
check(file_exists($statementsPath) && file_exists($chatPath), 'toy world corpora exist (run bin/build-toy-world.php)');

$chatLines = explode("\n", trim(file_get_contents($chatPath)));
$wellFormed = true;
foreach (array_slice($chatLines, 0, 200) as $line) {
    $wellFormed = $wellFormed
        && str_starts_with($line, '<user> ')
        && str_contains($line, ' <bot> ')
        && str_ends_with($line, ' <end>');
}
check($wellFormed, 'every dialogue follows the chat template: <user> … <bot> … <end>');
check(str_contains(file_get_contents($statementsPath), 'mia lives in the red house'), 'statements corpus contains the world facts');
check(!str_contains(file_get_contents($statementsPath), '<user>'), 'pretraining corpus contains NO chat tokens — the base model has never seen the format');

// ---- A tiny model can be fine-tuned to answer -------------------------------
$miniText = "<user> where is milo <bot> milo is in the kitchen <end>\n<user> where is rex <bot> rex is in the garden <end>";
$miniTokenizer = WordTokenizer::fromText($miniText);
$model = new GptLanguageModel($miniTokenizer->vocabularySize(), 12, 16, 2, 2, new RandomNumberGenerator(4));
$sequences = [];
foreach (explode("\n", $miniText) as $line) {
    $tokenIds = $miniTokenizer->encode($line);
    $sequences[] = [array_slice($tokenIds, 0, -1), array_slice($tokenIds, 1)];
}
for ($step = 0; $step < 250; $step++) {
    $model->zeroGradients();
    foreach ($sequences as [$inputIds, $targetIds]) {
        $model->computeLoss($inputIds, $targetIds)->backward();
    }
    $model->applyGradientStep(0.15 / count($sequences));
}
$sampler = new TokenSampler(0.0); // greedy: we want its single best answer
$prompt = $miniTokenizer->encode('<user> where is rex <bot>');
$answerIds = [];
$endTokenId = $miniTokenizer->encode('<end>')[0];
for ($i = 0; $i < 8; $i++) {
    $nextTokenId = $sampler->sampleFromLogits($model->nextTokenLogits($prompt), new RandomNumberGenerator(1));
    if ($nextTokenId === $endTokenId) {
        break;
    }
    $answerIds[] = $nextTokenId;
    $prompt[] = $nextTokenId;
}
$answer = $miniTokenizer->decode($answerIds);
check(str_contains($answer, 'garden'), sprintf('a fine-tuned mini-GPT answers in chat format ("%s")', $answer));

// ---- The real chatbot, if trained -------------------------------------------
if (file_exists(__DIR__ . '/../models/chatbot.json')) {
    $realTokenizer = WordTokenizer::fromText(
        file_get_contents($statementsPath) . ' ' . file_get_contents($chatPath)
    );
    $chatbot = GptLanguageModel::loadFromFile(__DIR__ . '/../models/chatbot.json');
    $prompt = $realTokenizer->encode('<user> where does mia live <bot>');
    $endTokenId = $realTokenizer->encode('<end>')[0];
    $answerIds = [];
    for ($i = 0; $i < 12; $i++) {
        $nextTokenId = (new TokenSampler(0.0))->sampleFromLogits($chatbot->nextTokenLogits($prompt), new RandomNumberGenerator(1));
        if ($nextTokenId === $endTokenId) {
            break;
        }
        $answerIds[] = $nextTokenId;
        $prompt[] = $nextTokenId;
    }
    check(str_contains($realTokenizer->decode($answerIds), 'red'), 'the trained chatbot knows where mia lives (greedy answer contains "red")');
} else {
    echo "· skipping trained-chatbot check (run bin/train-chatbot.php first)\n";
}

echo "\nAll checks passed. Talk to it: php bin/chat.php\n";
