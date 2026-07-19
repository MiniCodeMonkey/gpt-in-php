<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';

use Llm\Matrix;

/** Compare two matrices within a small tolerance. */
function assertMatEquals(array $expected, array $actual, string $label): void
{
    $ok = count($expected) === count($actual);
    if ($ok) {
        foreach ($expected as $i => $row) {
            if (count($row) !== count($actual[$i])) { $ok = false; break; }
            foreach ($row as $j => $v) {
                if (abs($v - $actual[$i][$j]) > 1e-9) { $ok = false; break 2; }
            }
        }
    }
    if (!$ok) {
        echo "✗ {$label}\n  expected: " . json_encode($expected) . "\n  got:      " . json_encode($actual) . "\n";
        exit(1);
    }
    echo "✓ {$label}\n";
}

assertMatEquals(
    [[19.0, 22.0], [43.0, 50.0]],
    Matrix::multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
    '2×2 · 2×2'
);

assertMatEquals(
    [[1.0, 2.0], [3.0, 4.0]],
    Matrix::multiply([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]),
    'multiplying by identity changes nothing'
);

assertMatEquals(
    [[58.0, 64.0], [139.0, 154.0]],
    Matrix::multiply([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    '2×3 · 3×2 (rectangular shapes)'
);

assertMatEquals(
    [[-2.5]],
    Matrix::multiply([[0.5, -1.0]], [[1.0], [3.0]]),
    '1×2 · 2×1 (a single dot product)'
);

echo "\nAll checks passed. Now benchmark:\n  php bin/bench.php\n  ./bin/phpj bin/bench.php\n";
