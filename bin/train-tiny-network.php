<?php

declare(strict_types=1);

require __DIR__ . '/../src/Value.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/TinyNeuralNetwork.php';

use Llm\RandomNumberGenerator;
use Llm\TinyNeuralNetwork;
use Llm\Value;

// Chapter 4 finale: watch gradient descent learn XOR.
//
// XOR — "output true when exactly one input is true" — is historically loaded:
// in 1969 Minsky & Papert proved no single-layer network can represent it,
// which helped freeze neural network research for a decade. One hidden layer
// and a nonlinearity dissolve the problem. You're about to reproduce, in PHP,
// the result that un-froze the field.

$random = new RandomNumberGenerator(1234);
$network = new TinyNeuralNetwork([2, 4, 1], $random);
printf("Network 2 → 4 → 1: %d learnable parameters (Claude has ~a trillion; the physics is identical)\n\n", count($network->parameters()));

$inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
$targets = [-1.0, 1.0, 1.0, -1.0]; // tanh outputs live in (-1, 1): false = -1, true = +1

$show = function (TinyNeuralNetwork $network) use ($inputs, $targets): void {
    foreach ($inputs as $index => $input) {
        $prediction = $network->forward($input)->data;
        printf("  XOR(%d, %d) = %+.3f   (want %+d)  %s\n",
            (int) $input[0], (int) $input[1], $prediction, (int) $targets[$index],
            ($prediction > 0) === ($targets[$index] > 0) ? 'right sign' : 'WRONG');
    }
};

echo "Before training (weights are random):\n";
$show($network);

$learningRate = 0.1;
echo "\nTraining: forward → measure loss → backward → nudge every knob downhill\n";
for ($step = 1; $step <= 500; $step++) {
    // Forward: run all four cases, sum the squared errors into one loss Value.
    $totalLoss = new Value(0.0);
    foreach ($inputs as $index => $input) {
        $prediction = $network->forward($input);
        $totalLoss = $totalLoss->add($prediction->subtract($targets[$index])->power(2.0));
    }

    // Backward: fill every parameter's ->gradient in one pass.
    $network->zeroGradients();
    $totalLoss->backward();

    // Descend: move each knob a small step AGAINST its gradient.
    foreach ($network->parameters() as $parameter) {
        $parameter->data -= $learningRate * $parameter->gradient;
    }

    if ($step === 1 || $step % 100 === 0) {
        printf("  step %3d   loss %.4f\n", $step, $totalLoss->data);
    }
}

echo "\nAfter training (same knobs, new values):\n";
$show($network);
echo "\nNothing was programmed. 17 numbers were nudged 500 times, each time in the\ndirection backward() said would reduce the loss. That is all \"learning\" is.\n";
