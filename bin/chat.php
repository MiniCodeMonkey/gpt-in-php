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

// The finale: php bin/chat.php — talk to the model you built from scratch.

$tokenizer = WordTokenizer::fromText(
    file_get_contents(__DIR__ . '/../data/toy-world-statements.txt') . ' ' .
    file_get_contents(__DIR__ . '/../data/toy-world-chat.txt')
);
$model = GptLanguageModel::loadFromFile(__DIR__ . '/../models/chatbot.json');
$sampler = new TokenSampler(temperature: 0.3);
$endTokenId = $tokenizer->encode('<end>')[0];

echo "── GPT in PHP — toy world chat ─────────────────────────────\n";
echo "It knows 5 people: mia, leo, sam, ana, max — their houses,\n";
echo "pets, foods, and jobs. Ask things like \"where does mia live\"\n";
echo "or \"who is rex\". Ctrl-C or empty line to quit.\n\n";

$conversationSeed = 1;
while (true) {
    $line = readline('you: ');
    if ($line === false || trim($line) === '') {
        echo "bye!\n";
        break;
    }
    $question = strtolower(trim(preg_replace('/[^a-zA-Z\s]/', '', $line)));

    $unknownWords = array_filter(
        preg_split('/\s+/', $question) ?: [],
        fn (string $word) => $word !== '' && !$tokenizer->knows($word)
    );
    if ($unknownWords !== []) {
        printf("bot: (my whole world is %d words — I don't know \"%s\")\n\n",
            $tokenizer->vocabularySize(), implode('", "', $unknownWords));
        continue;
    }

    $random = new RandomNumberGenerator($conversationSeed++);
    $tokenIds = $tokenizer->encode("<user> {$question} <bot>");
    $answerIds = [];
    for ($i = 0; $i < 16; $i++) {
        $window = array_slice($tokenIds, -$model->contextLength);
        $nextTokenId = $sampler->sampleFromLogits($model->nextTokenLogits($window), $random);
        if ($nextTokenId === $endTokenId) {
            break;
        }
        $answerIds[] = $nextTokenId;
        $tokenIds[] = $nextTokenId;
    }
    printf("bot: %s\n\n", $tokenizer->decode($answerIds));
}
