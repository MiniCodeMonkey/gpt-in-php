<?php

declare(strict_types=1);

require __DIR__ . '/../src/Value.php';
require __DIR__ . '/../src/RandomNumberGenerator.php';
require __DIR__ . '/../src/TinyNeuralNetwork.php';

use Llm\RandomNumberGenerator;
use Llm\TinyNeuralNetwork;
use Llm\Value;

function check(bool $condition, string $label): void
{
    if (!$condition) {
        echo "✗ {$label}\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

// ---- Hand-verifiable gradients ---------------------------------------------
// L = (a + b) * c with a=2, b=3, c=4  →  L = 20
// Nudge a: L = (2.001+3)*4 = 20.004 → gradient of a is 4 (= c). Same for b. Gradient of c is 5 (= a+b).
$a = new Value(2.0);
$b = new Value(3.0);
$c = new Value(4.0);
$loss = $a->add($b)->multiply($c);
$loss->backward();

check($loss->data === 20.0, 'forward pass: (2 + 3) × 4 = 20');
check($a->gradient === 4.0 && $b->gradient === 4.0, 'gradient through add: both sides inherit ×c = 4');
check($c->gradient === 5.0, 'gradient through multiply: the other factor, a+b = 5');

// ---- The classic bug: a value used twice must ACCUMULATE gradient ----------
// L = a × a with a=3 → dL/da = 2a = 6 (each use contributes 3; += makes it 6)
$a = new Value(3.0);
$loss = $a->multiply($a);
$loss->backward();
check($a->gradient === 6.0, 'a value used twice accumulates gradient from both uses (d(a²)/da = 2a)');

// ---- tanh --------------------------------------------------------------------
$x = new Value(0.5);
$y = $x->tanh();
$y->backward();
check(abs($x->gradient - (1.0 - tanh(0.5) ** 2)) < 1e-12, 'tanh gradient is 1 − tanh²(x)');

// ---- The star check: analytic gradients vs numerically nudging the inputs --
// f(x, y) = tanh(x·y + x²) · e^(y/2) + log(x) — ugly on purpose.
$compute = function (float $xValue, float $yValue): array {
    $x = new Value($xValue);
    $y = new Value($yValue);
    $f = $x->multiply($y)->add($x->power(2.0))->tanh()
        ->multiply($y->multiply(0.5)->exponential())
        ->add($x->logarithm());

    return [$f, $x, $y];
};

[$f, $x, $y] = $compute(0.7, -1.3);
$f->backward();

$nudge = 1e-6;
$numericalX = ($compute(0.7 + $nudge, -1.3)[0]->data - $compute(0.7 - $nudge, -1.3)[0]->data) / (2 * $nudge);
$numericalY = ($compute(0.7, -1.3 + $nudge)[0]->data - $compute(0.7, -1.3 - $nudge)[0]->data) / (2 * $nudge);

check(abs($x->gradient - $numericalX) < 1e-6, sprintf('backprop matches physically nudging x (%.7f vs %.7f)', $x->gradient, $numericalX));
check(abs($y->gradient - $numericalY) < 1e-6, sprintf('backprop matches physically nudging y (%.7f vs %.7f)', $y->gradient, $numericalY));

// ---- The payoff: a neural network learns XOR --------------------------------
// XOR is famous: no single-layer network can learn it (it's not linearly
// separable). A hidden layer + nonlinearity cracks it — this experiment ended
// the first AI winter's central objection to neural networks.
$random = new RandomNumberGenerator(1234);
$network = new TinyNeuralNetwork([2, 4, 1], $random);

$inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
$targets = [-1.0, 1.0, 1.0, -1.0]; // tanh speaks in (-1, 1), so "false" is -1

$firstLoss = null;
$lastLoss = null;
for ($step = 0; $step < 500; $step++) {
    $totalLoss = new Value(0.0);
    foreach ($inputs as $index => $input) {
        $prediction = $network->forward($input);
        $totalLoss = $totalLoss->add($prediction->subtract($targets[$index])->power(2.0));
    }
    $network->zeroGradients();
    $totalLoss->backward();
    foreach ($network->parameters() as $parameter) {
        $parameter->data -= 0.1 * $parameter->gradient; // gradient descent: step downhill
    }
    $firstLoss ??= $totalLoss->data;
    $lastLoss = $totalLoss->data;
}

check($lastLoss < 0.05, sprintf('gradient descent learns XOR (loss %.3f → %.4f over 500 steps)', $firstLoss, $lastLoss));

$correct = true;
foreach ($inputs as $index => $input) {
    $prediction = $network->forward($input)->data;
    $correct = $correct && (($prediction > 0) === ($targets[$index] > 0));
}
check($correct, 'all four XOR cases predicted with the right sign');

echo "\nAll checks passed. Watch the training run: php bin/train-tiny-network.php\n";
