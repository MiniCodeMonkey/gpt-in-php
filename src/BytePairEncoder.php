<?php

declare(strict_types=1);

namespace Llm;

/**
 * A miniature byte-pair encoder (BPE) — the algorithm behind real LLM tokenizers.
 *
 * Idea: start from single characters, then repeatedly find the most frequent
 * adjacent pair of tokens in the training text and glue it into one new token.
 * After a few thousand merges you get tokens like "the", "ing", " Copenhagen" —
 * a vocabulary shaped by the data itself.
 *
 * Ours works on characters within words (real ones work on raw bytes across
 * everything), which keeps the algorithm identical but easy to watch.
 */
final class BytePairEncoder
{
    /**
     * @param array<int, array{string, string}> $orderedMerges each learned merge,
     *        in the order learned — order matters when encoding new text
     */
    private function __construct(
        private readonly array $orderedMerges,
    ) {
    }

    /**
     * Learn $numberOfMerges merges from the words in $text (split on whitespace).
     */
    public static function train(string $text, int $numberOfMerges): self
    {
        // Each distinct word starts as a list of single-character tokens.
        // We keep a count per word so frequent words weigh more.
        $wordCounts = array_count_values(preg_split('/\s+/', trim($text)) ?: []);
        $wordTokens = [];
        foreach ($wordCounts as $word => $count) {
            $wordTokens[$word] = str_split((string) $word);
        }

        $orderedMerges = [];
        for ($mergeIndex = 0; $mergeIndex < $numberOfMerges; $mergeIndex++) {
            // Count every adjacent token pair across the whole corpus.
            $pairFrequencies = [];
            foreach ($wordTokens as $word => $tokens) {
                $weight = $wordCounts[$word];
                for ($i = 0; $i < count($tokens) - 1; $i++) {
                    $pairKey = $tokens[$i] . "\u{1F}" . $tokens[$i + 1]; // unit separator: never appears in text
                    $pairFrequencies[$pairKey] = ($pairFrequencies[$pairKey] ?? 0) + $weight;
                }
            }
            if ($pairFrequencies === []) {
                break; // every word is a single token already — nothing left to merge
            }

            arsort($pairFrequencies);
            $bestPairKey = array_key_first($pairFrequencies);
            [$first, $second] = explode("\u{1F}", $bestPairKey);
            $orderedMerges[] = [$first, $second];

            // Apply the merge everywhere before looking for the next one.
            foreach ($wordTokens as $word => $tokens) {
                $wordTokens[$word] = self::applyMerge($tokens, $first, $second);
            }
        }

        return new self($orderedMerges);
    }

    /**
     * Tokenize one word by replaying the learned merges in training order.
     *
     * @return array<int, string> the word's tokens, e.g. "isabella" → ["is", "a", "bella"]
     */
    public function tokenizeWord(string $word): array
    {
        $tokens = str_split($word);
        foreach ($this->orderedMerges as [$first, $second]) {
            $tokens = self::applyMerge($tokens, $first, $second);
        }

        return $tokens;
    }

    /** @return array<int, array{string, string}> */
    public function merges(): array
    {
        return $this->orderedMerges;
    }

    /**
     * Replace every adjacent [$first, $second] in $tokens with the glued token.
     *
     * @param array<int, string> $tokens
     * @return array<int, string>
     */
    private static function applyMerge(array $tokens, string $first, string $second): array
    {
        $result = [];
        $i = 0;
        $count = count($tokens);
        while ($i < $count) {
            if ($i < $count - 1 && $tokens[$i] === $first && $tokens[$i + 1] === $second) {
                $result[] = $first . $second;
                $i += 2;
            } else {
                $result[] = $tokens[$i];
                $i += 1;
            }
        }

        return $result;
    }
}
