<?php

declare(strict_types=1);

namespace Llm;

/**
 * A deterministic random number generator (xorshift32).
 *
 * Why not mt_rand()? Because everything in this course must be reproducible:
 * the same seed must produce the same "random" names, the same initial weights,
 * the same training run — on any machine, any PHP version. Real ML frameworks
 * obsess over seeding for exactly this reason.
 *
 * xorshift32 needs only XORs and bit shifts, so it behaves identically on every
 * platform (no overflow subtleties), while being plenty random for our needs.
 */
final class RandomNumberGenerator
{
    private int $state;

    public function __construct(int $seed)
    {
        $this->state = $seed & 0xFFFFFFFF;
        if ($this->state === 0) {
            $this->state = 0x9E3779B9; // xorshift must never start at zero
        }
    }

    /** A float in [0, 1) — the building block for everything else. */
    public function nextFloat(): float
    {
        $s = $this->state;
        $s = ($s ^ ($s << 13)) & 0xFFFFFFFF;
        $s = $s ^ ($s >> 17);
        $s = ($s ^ ($s << 5)) & 0xFFFFFFFF;
        $this->state = $s;

        return $s / 4294967296.0; // divide by 2^32
    }

    /**
     * Draw one index from a probability distribution — the "sampling" step of
     * every language model. $probabilities must sum to 1, e.g. [0.5, 0.3, 0.2]
     * returns 0 half the time, 1 thirty percent of the time, 2 the rest.
     *
     * How: imagine the interval [0,1) split into segments sized like the
     * probabilities; draw a random point; return which segment it landed in.
     *
     * @param array<int, float> $probabilities
     */
    public function sampleFromDistribution(array $probabilities): int
    {
        $point = $this->nextFloat();
        $cumulative = 0.0;
        foreach ($probabilities as $index => $probability) {
            $cumulative += $probability;
            if ($point < $cumulative) {
                return $index;
            }
        }

        return array_key_last($probabilities); // guard against float rounding at the far end
    }
}
