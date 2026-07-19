<?php

declare(strict_types=1);

require __DIR__ . '/../src/Matrix.php';

use Llm\Matrix;

// Chapter 1 benchmark: how many floating-point multiply-adds per second
// can this machine do through Matrix::multiply?
//
// One n×n·n×n matmul costs 2·n³ floating-point operations (a multiply and
// an add per inner step). We time repeated 128×128 multiplies and report MFLOP/s.

$jit = 'off';
if (function_exists('opcache_get_status')) {
    $s = @opcache_get_status(false);
    if (!empty($s['jit']['on']) || !empty($s['jit']['enabled'])) {
        $jit = 'ON';
    }
}
printf("PHP %s — JIT %s\n", PHP_VERSION, $jit);

mt_srand(42); // fixed seed: same matrices every run
$n = 128;
$mk = fn () => array_map(
    fn () => array_map(fn () => mt_rand() / mt_getrandmax() - 0.5, range(1, $n)),
    range(1, $n)
);
$a = $mk();
$b = $mk();

for ($w = 0; $w < 3; $w++) {
    Matrix::multiply($a, $b); // warm-up (lets the JIT trace and compile the hot loop before we time it)
}

$reps = 20;
$t0 = hrtime(true);
for ($i = 0; $i < $reps; $i++) {
    $c = Matrix::multiply($a, $b);
}
$seconds = (hrtime(true) - $t0) / 1e9;

$flops = 2 * $n ** 3 * $reps;
printf(
    "%d × (%d×%d matmul) in %.2fs  →  %.1f MFLOP/s\n",
    $reps, $n, $n, $seconds, $flops / $seconds / 1e6
);
printf("(checksum %.6f — should match across runs)\n", $c[0][0]);
