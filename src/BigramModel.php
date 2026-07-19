<?php

declare(strict_types=1);

namespace Llm;

/**
 * The bigram model: our first generative language model, built from counting.
 *
 * It answers exactly one question: given the current character, what is the
 * probability of each possible next character? Training is literally tallying —
 * walk the dataset and count every "this character was followed by that one".
 *
 * The newline token (ID 0) does double duty as a boundary marker: a name is
 * modeled as \n → m → a → t → … → \n, so the model learns both which characters
 * *start* names (what follows \n) and which characters *end* them (what \n follows).
 */
final class BigramModel
{
    /**
     * @param array<int, array<int, float>> $transitionProbabilities row = current
     *        token ID, column = next token ID, each row summing to 1
     * @param array<int, array<int, int>> $transitionCounts the raw tallies
     */
    private function __construct(
        private readonly array $transitionProbabilities,
        private readonly array $transitionCounts,
        private readonly CharacterTokenizer $tokenizer,
    ) {
    }

    /**
     * "Training": count every adjacent pair of tokens, then turn each row of
     * counts into probabilities by dividing by the row's total.
     *
     * $smoothing adds a small phantom count to every possible pair, so that a
     * pair never seen in training gets a tiny probability instead of zero.
     * Without it, one unseen pair in the test data would make the model's loss
     * infinite (log of zero) — a real technique with a 250-year history
     * (Laplace smoothing).
     *
     * @param array<int, string> $names
     */
    public static function trainOn(array $names, CharacterTokenizer $tokenizer, float $smoothing = 1.0): self
    {
        $vocabularySize = $tokenizer->vocabularySize();

        $counts = array_fill(0, $vocabularySize, array_fill(0, $vocabularySize, 0));
        foreach ($names as $name) {
            $tokenIds = [0, ...$tokenizer->encode($name), 0]; // wrap in boundary markers
            for ($i = 0; $i < count($tokenIds) - 1; $i++) {
                $counts[$tokenIds[$i]][$tokenIds[$i + 1]]++;
            }
        }

        $probabilities = [];
        foreach ($counts as $currentTokenId => $row) {
            $rowTotal = array_sum($row) + $smoothing * $vocabularySize;
            foreach ($row as $nextTokenId => $count) {
                $probabilities[$currentTokenId][$nextTokenId] = ($count + $smoothing) / $rowTotal;
            }
        }

        return new self($probabilities, $counts, $tokenizer);
    }

    /**
     * Generate one name: start at the boundary token, repeatedly sample the next
     * character from the current character's probability row, stop when the
     * boundary token is drawn again. This loop — look up distribution, sample,
     * append, repeat — is the same loop ChatGPT runs, just with a better model.
     */
    public function generate(RandomNumberGenerator $random, int $maximumLength = 24): string
    {
        $currentTokenId = 0;
        $generatedTokenIds = [];
        while (count($generatedTokenIds) < $maximumLength) {
            $nextTokenId = $random->sampleFromDistribution($this->transitionProbabilities[$currentTokenId]);
            if ($nextTokenId === 0) {
                break; // the model chose to end the name
            }
            $generatedTokenIds[] = $nextTokenId;
            $currentTokenId = $nextTokenId;
        }

        return $this->tokenizer->decode($generatedTokenIds);
    }

    /**
     * The loss: average negative log-likelihood per character.
     *
     * For every transition that actually occurs in $names, look up the
     * probability the model assigned to it, take the log, negate, average.
     * A model that finds the real data likely scores low; one that is
     * "surprised" by the data scores high. A model with no knowledge at all
     * (uniform over 27 tokens) scores exactly ln(27) ≈ 3.296 — our baseline.
     *
     * This exact quantity, under the name cross-entropy, is what every neural
     * network in this course will be trained to minimize.
     *
     * @param array<int, string> $names
     */
    public function averageNegativeLogLikelihood(array $names): float
    {
        $totalNegativeLogLikelihood = 0.0;
        $transitionCount = 0;
        foreach ($names as $name) {
            $tokenIds = [0, ...$this->tokenizer->encode($name), 0];
            for ($i = 0; $i < count($tokenIds) - 1; $i++) {
                $probability = $this->transitionProbabilities[$tokenIds[$i]][$tokenIds[$i + 1]];
                $totalNegativeLogLikelihood += -log($probability);
                $transitionCount++;
            }
        }

        return $totalNegativeLogLikelihood / $transitionCount;
    }

    /** @return array<int, array<int, float>> */
    public function transitionProbabilities(): array
    {
        return $this->transitionProbabilities;
    }

    /** @return array<int, array<int, int>> */
    public function transitionCounts(): array
    {
        return $this->transitionCounts;
    }
}
