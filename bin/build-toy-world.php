<?php

declare(strict_types=1);

require __DIR__ . '/../src/RandomNumberGenerator.php';

use Llm\RandomNumberGenerator;

// Chapter 10, step 1: create the toy world — the TinyStories trick.
//
// A 40k-parameter model can't be coherent about the open world; it CAN be
// coherent about a world with 5 people and 20 facts. We generate two corpora
// from the same fact table:
//
//   data/toy-world-statements.txt — plain sentences (the "internet" of this
//     world; what PRETRAINING reads)
//   data/toy-world-chat.txt — the same knowledge as <user>/<bot> dialogues
//     (what FINE-TUNING reads)

$world = [
    'mia' => ['house' => 'red', 'pet' => 'cat', 'petName' => 'milo', 'food' => 'pizza', 'job' => 'baker'],
    'leo' => ['house' => 'blue', 'pet' => 'dog', 'petName' => 'rex', 'food' => 'apples', 'job' => 'teacher'],
    'sam' => ['house' => 'green', 'pet' => 'bird', 'petName' => 'kiwi', 'food' => 'soup', 'job' => 'doctor'],
    'ana' => ['house' => 'yellow', 'pet' => 'fish', 'petName' => 'bubbles', 'food' => 'cake', 'job' => 'farmer'],
    'max' => ['house' => 'white', 'pet' => 'pony', 'petName' => 'star', 'food' => 'tea', 'job' => 'painter'],
];

// ---- Statements: every fact, said several ways ------------------------------
$statements = [];
foreach ($world as $person => $facts) {
    $statements[] = "{$person} lives in the {$facts['house']} house";
    $statements[] = "the {$facts['house']} house is where {$person} lives";
    $statements[] = "{$person} has a {$facts['pet']} named {$facts['petName']}";
    $statements[] = "{$facts['petName']} is the {$facts['pet']} of {$person}";
    $statements[] = "{$person} likes {$facts['food']}";
    $statements[] = "{$person} likes to eat {$facts['food']}";
    $statements[] = "{$person} works as a {$facts['job']}";
    $statements[] = "the {$facts['job']} is {$person}";
}

// ---- Dialogues: every fact, asked several ways ------------------------------
$dialogues = [];
foreach ($world as $person => $facts) {
    $answerHouse = "{$person} lives in the {$facts['house']} house";
    $answerPet = "{$person} has a {$facts['pet']} named {$facts['petName']}";
    $answerFood = "{$person} likes {$facts['food']}";
    $answerJob = "{$person} works as a {$facts['job']}";

    foreach ([
        ["where does {$person} live", $answerHouse],
        ["who lives in the {$facts['house']} house", $answerHouse],
        ["what pet does {$person} have", $answerPet],
        ["who is {$facts['petName']}", "{$facts['petName']} is the {$facts['pet']} of {$person}"],
        ["what does {$person} like to eat", $answerFood],
        ["who likes {$facts['food']}", $answerFood],
        ["what does {$person} work as", $answerJob],
        ["who works as a {$facts['job']}", $answerJob],
    ] as [$question, $answer]) {
        $dialogues[] = "<user> {$question} <bot> {$answer} <end>";
    }
}

// Repeat and shuffle (seeded) so training sees varied order, then write.
$random = new RandomNumberGenerator(99);
$shuffleAndRepeat = function (array $lines, int $repetitions) use ($random): array {
    $repeated = [];
    for ($i = 0; $i < $repetitions; $i++) {
        $copy = $lines;
        for ($j = count($copy) - 1; $j > 0; $j--) {
            $k = (int) floor($random->nextFloat() * ($j + 1));
            [$copy[$j], $copy[$k]] = [$copy[$k], $copy[$j]];
        }
        $repeated = [...$repeated, ...$copy];
    }

    return $repeated;
};

file_put_contents(__DIR__ . '/../data/toy-world-statements.txt', implode("\n", $shuffleAndRepeat($statements, 40)) . "\n");
file_put_contents(__DIR__ . '/../data/toy-world-chat.txt', implode("\n", $shuffleAndRepeat($dialogues, 20)) . "\n");

printf("World: %d people, %d facts\n", count($world), count($world) * 4);
printf("Wrote %d statement lines (%d distinct) and %d dialogue lines (%d distinct)\n",
    count($statements) * 40, count($statements), count($dialogues) * 20, count($dialogues));
