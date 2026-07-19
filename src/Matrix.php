<?php

declare(strict_types=1);

namespace Llm;

/**
 * Matrix operations on plain PHP arrays-of-rows, e.g. [[1.0, 2.0], [3.0, 4.0]].
 *
 * This file grows over the course. Chapter 1: multiply.
 */
final class Matrix
{
    /**
     * Matrix multiplication: returns the product of $left (n×k) and $right (k×m),
     * an n×m matrix where every cell is a row-times-column dot product:
     *
     *   product[row][column] = Σ over inner of left[row][inner] * right[inner][column]
     *
     * This single operation will consume ~95% of the model's training time by
     * chapter 8 — every layer of a neural network is built on it.
     * (The ML world calls this "matmul".)
     */
    public static function multiply(array $left, array $right): array
    {
        $rowCount = count($left);          // n: rows in the result
        $innerCount = count($right);       // k: the shared dimension that must match
        $columnCount = count($right[0]);   // m: columns in the result

        if (count($left[0]) !== $innerCount) {
            throw new \InvalidArgumentException(sprintf(
                'Shape mismatch: left is %d×%d but right is %d×%d — inner dimensions must match',
                $rowCount, count($left[0]), $innerCount, $columnCount
            ));
        }

        $product = [];
        for ($row = 0; $row < $rowCount; $row++) {
            $leftRow = $left[$row]; // hoisted out of the inner loops: one array lookup instead of three
            for ($column = 0; $column < $columnCount; $column++) {
                $sum = 0.0;
                for ($inner = 0; $inner < $innerCount; $inner++) {
                    $sum += $leftRow[$inner] * $right[$inner][$column];
                }
                $product[$row][$column] = $sum;
            }
        }

        return $product;
    }
}
