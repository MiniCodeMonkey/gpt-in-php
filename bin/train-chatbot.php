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

// Chapter 10, step 2: the frontier pipeline in miniature.
//
//   Stage 1 — PRETRAINING: the GPT reads this world's "internet" (plain
//     statements) and learns its language and facts. Result: a text completer.
//   Stage 2 — SUPERVISED FINE-TUNING: the SAME weights continue training on
//     <user>/<bot> dialogues. Result: a chatbot.
//
// This split is exactly how Claude and ChatGPT are made (plus a chapter-11
// step we can only read about at this scale: reinforcement learning).

$statementsText = file_get_contents(__DIR__ . '/../data/toy-world-statements.txt');
$chatText = file_get_contents(__DIR__ . '/../data/toy-world-chat.txt');

// One tokenizer over BOTH corpora, so <user>/<bot>/<end> have IDs from day one
// (real labs also reserve special tokens before pretraining).
$tokenizer = WordTokenizer::fromText($statementsText . ' ' . $chatText);
printf("Vocabulary: %d words (including <user>, <bot>, <end>)\n", $tokenizer->vocabularySize());

$contextLength = 20;
$toSequences = function (string $text) use ($tokenizer, $contextLength): array {
    $sequences = [];
    foreach (explode("\n", trim($text)) as $line) {
        $tokenIds = array_slice($tokenizer->encode($line), 0, $contextLength + 1);
        if (count($tokenIds) >= 2) {
            $sequences[] = [array_slice($tokenIds, 0, -1), array_slice($tokenIds, 1)];
        }
    }

    return $sequences;
};
$statementSequences = $toSequences($statementsText);
$chatSequences = $toSequences($chatText);

$random = new RandomNumberGenerator(2026);
$model = new GptLanguageModel(
    vocabularySize: $tokenizer->vocabularySize(),
    contextLength: $contextLength,
    embeddingDimensions: 32,
    headCount: 4,
    blockCount: 2,
    random: $random,
);
printf("Model: same GPT class as chapter 8, %s parameters\n\n", number_format($model->parameterCount()));

$trainOn = function (array $sequences, int $totalSteps, string $label) use ($model, $random): void {
    $sequenceCount = count($sequences);
    $sequencesPerStep = 24;
    $startedAt = hrtime(true);
    for ($step = 1; $step <= $totalSteps; $step++) {
        $learningRate = ($step <= (int) ($totalSteps * 0.8) ? 0.3 : 0.06) / $sequencesPerStep;
        $model->zeroGradients();
        $batchLoss = 0.0;
        for ($i = 0; $i < $sequencesPerStep; $i++) {
            [$inputIds, $targetIds] = $sequences[(int) floor($random->nextFloat() * $sequenceCount)];
            $loss = $model->computeLoss($inputIds, $targetIds);
            $loss->backward();
            $batchLoss += $loss->data[0][0] / $sequencesPerStep;
        }
        $model->applyGradientStep($learningRate);
        if ($step % 250 === 0 || $step === 1) {
            printf("  [%s] step %4d   batch loss %.4f   (%.0fs)\n", $label, $step, $batchLoss, (hrtime(true) - $startedAt) / 1e9);
        }
    }
};

$respond = function (string $question, float $temperature = 0.3) use ($model, $tokenizer): string {
    $sampler = new TokenSampler($temperature);
    $random = new RandomNumberGenerator(7);
    $tokenIds = $tokenizer->encode("<user> {$question} <bot>");
    $answerIds = [];
    $endTokenId = $tokenizer->encode('<end>')[0];
    for ($i = 0; $i < 16; $i++) {
        $window = array_slice($tokenIds, -$model->contextLength);
        $nextTokenId = $sampler->sampleFromLogits($model->nextTokenLogits($window), $random);
        if ($nextTokenId === $endTokenId) {
            break;
        }
        $answerIds[] = $nextTokenId;
        $tokenIds[] = $nextTokenId;
    }

    return $tokenizer->decode($answerIds);
};

// ---- Stage 1: pretraining ---------------------------------------------------
echo "STAGE 1 — pretraining on the world's plain text:\n";
$trainOn($statementSequences, 1500, 'pretrain');
$model->saveToFile(__DIR__ . '/../models/chatbot-base.json');

echo "\nThe BASE model, asked a question (chat format it has never seen):\n";
printf("  <user> where does mia live <bot> → \"%s\"\n", $respond('where does mia live'));
echo "…knows the facts but not the format. Now teach it to converse:\n\n";

// ---- Stage 2: supervised fine-tuning ----------------------------------------
echo "STAGE 2 — fine-tuning the SAME weights on dialogues:\n";
$trainOn($chatSequences, 2000, 'finetune');
$model->saveToFile(__DIR__ . '/../models/chatbot.json');

// ---- Evaluate: ask every fact, canonical phrasing ---------------------------
$world = [
    'mia' => ['red', 'milo', 'pizza', 'baker'],
    'leo' => ['blue', 'rex', 'apples', 'teacher'],
    'sam' => ['green', 'kiwi', 'soup', 'doctor'],
    'ana' => ['yellow', 'bubbles', 'cake', 'farmer'],
    'max' => ['white', 'star', 'tea', 'painter'],
];
$correct = 0;
$total = 0;
echo "\nExam — one question per fact (answer must contain the key detail):\n";
foreach ($world as $person => [$house, $petName, $food, $job]) {
    foreach ([
        ["where does {$person} live", $house],
        ["what pet does {$person} have", $petName],
        ["what does {$person} like to eat", $food],
        ["what does {$person} work as", $job],
    ] as [$question, $mustContain]) {
        $answer = $respond($question);
        $isCorrect = str_contains($answer, $mustContain);
        $correct += $isCorrect ? 1 : 0;
        $total++;
        if (!$isCorrect) {
            printf("  ✗ %-32s → %s (wanted: %s)\n", $question, $answer, $mustContain);
        }
    }
}
printf("Score: %d/%d facts answered correctly\n", $correct, $total);

echo "\nSample conversation with the FINE-TUNED model:\n";
foreach (['where does ana live', 'who is rex', 'what does max work as', 'who likes soup'] as $question) {
    printf("  you: %s\n  bot: %s\n", $question, $respond($question));
}

// Export a few before/after pairs for the textbook.
file_put_contents(__DIR__ . '/../docs/chatbot-results.json', json_encode([
    'score' => "{$correct}/{$total}",
    'vocabularySize' => $tokenizer->vocabularySize(),
    'parameterCount' => $model->parameterCount(),
]));
echo "\nSaved models/chatbot-base.json + models/chatbot.json. Talk to it: php bin/chat.php\n";
