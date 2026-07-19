<?php

declare(strict_types=1);

// Chapter 11: the arithmetic of the frontier, using YOUR Mac's measured speed.

$phpFlopsPerSecond = 1217e6; // measured in chapter 1: 1,217 MFLOP/s with JIT

$trainingRuns = [
    ['our names GPT (measured)', 6 * 27355 * 5000 * 32 * 12, null],
    ['GPT-2 (2019)', 8.6e20, '124M params'],
    ['GPT-3 (2020)', 3.1e23, '175B params'],
    ['frontier model (est. 2025)', 3e25, 'undisclosed'],
];

// Rough rule (Kaplan/Chinchilla): training FLOPs ≈ 6 × parameters × tokens.
echo "If this Mac's PHP had to train each model (at your measured 1,217 MFLOP/s):\n\n";
printf("  %-28s %-14s %s\n", 'model', 'FLOPs', 'wall-clock on this Mac');
foreach ($trainingRuns as [$name, $flops, $note]) {
    $seconds = $flops / $phpFlopsPerSecond;
    $display = match (true) {
        $seconds < 3600 => sprintf('%.0f minutes', $seconds / 60),
        $seconds < 86400 * 365 => sprintf('%.1f days', $seconds / 86400),
        $seconds < 86400 * 365 * 1e9 => sprintf('%s years', number_format($seconds / (86400 * 365))),
        default => sprintf('%.1f BILLION years (universe: 13.8)', $seconds / (86400 * 365 * 1e9)),
    };
    printf("  %-28s %-14s %s%s\n", $name, sprintf('%.1e', $flops), $display, $note ? "  ({$note})" : '');
}

echo "\nWhy frontier training is possible at all:\n";
$h100 = 1e15;
printf("  one H100 GPU ≈ %.0e FLOP/s ≈ %s× this PHP\n", $h100, number_format($h100 / $phpFlopsPerSecond));
printf("  a 20,000-GPU cluster ≈ %.0e FLOP/s → the frontier run above ≈ %.0f days\n",
    20000 * $h100 * 0.4, 3e25 / (20000 * $h100 * 0.4) / 86400);
echo "\nSame equations. Same gradients. Different electricity bill.\n";
