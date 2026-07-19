<?php

declare(strict_types=1);

namespace Llm;

/**
 * Matrix operations on plain PHP arrays-of-rows, e.g. [[1.0, 2.0], [3.0, 4.0]].
 *
 * This file grows over the course. Chapter 1: matmul.
 */
final class Mat
{
    /**
     * Matrix multiply: C = A · B.
     *
     * $a is n×k (n rows of k floats), $b is k×m. Returns the n×m result.
     * Definition: C[i][j] = sum over k of A[i][k] * B[k][j].
     *
     * TODO(you, chapter 1): implement with three nested loops.
     * Verify with: php tests/check-ch1.php
     */
    public static function matmul(array $a, array $b): array
    {
        throw new \RuntimeException('TODO: implement matmul (chapter 1)');
    }
}
