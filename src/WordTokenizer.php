<?php

declare(strict_types=1);

namespace Llm;

/**
 * A word-level tokenizer for the chatbot: every distinct whitespace-separated
 * word is one token. At our scale this buys enormous coherence — the model
 * predicts whole words instead of spelling letter by letter — at the price
 * that covers chapter 2's trade-off: a bigger vocabulary, and any word not in
 * it is unspeakable. Special tokens like <user> and <bot> are just words.
 */
final class WordTokenizer
{
    /**
     * @param array<int, string> $idToWord
     * @param array<string, int> $wordToId
     */
    private function __construct(
        private readonly array $idToWord,
        private readonly array $wordToId,
    ) {
    }

    public static function fromText(string $text): self
    {
        $words = preg_split('/\s+/', trim($text)) ?: [];
        $distinctWords = array_values(array_unique($words));
        sort($distinctWords);

        return new self($distinctWords, array_flip($distinctWords));
    }

    public function knows(string $word): bool
    {
        return isset($this->wordToId[$word]);
    }

    /** @return array<int, int> */
    public function encode(string $text): array
    {
        $tokenIds = [];
        foreach (preg_split('/\s+/', trim($text)) ?: [] as $word) {
            if ($word === '') {
                continue;
            }
            if (!isset($this->wordToId[$word])) {
                throw new \InvalidArgumentException("Word \"{$word}\" is not in the vocabulary.");
            }
            $tokenIds[] = $this->wordToId[$word];
        }

        return $tokenIds;
    }

    /** @param array<int, int> $tokenIds */
    public function decode(array $tokenIds): string
    {
        return implode(' ', array_map(fn (int $tokenId) => $this->idToWord[$tokenId], $tokenIds));
    }

    public function vocabularySize(): int
    {
        return count($this->idToWord);
    }

    /** @return array<int, string> */
    public function vocabulary(): array
    {
        return $this->idToWord;
    }
}
