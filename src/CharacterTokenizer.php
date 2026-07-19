<?php

declare(strict_types=1);

namespace Llm;

/**
 * A character-level tokenizer: every distinct character in the training text
 * becomes one token, numbered alphabetically from zero.
 *
 * The model never sees text — only these integer token IDs. The tokenizer is
 * the bridge, and encode/decode must be perfect inverses of each other.
 */
final class CharacterTokenizer
{
    /**
     * @param array<int, string> $idToCharacter e.g. [0 => "\n", 1 => 'a', 2 => 'b', …]
     * @param array<string, int> $characterToId the exact reverse mapping
     */
    private function __construct(
        private readonly array $idToCharacter,
        private readonly array $characterToId,
    ) {
    }

    /** Build the vocabulary from every distinct character found in $text. */
    public static function fromText(string $text): self
    {
        $distinctCharacters = array_values(array_unique(str_split($text)));
        sort($distinctCharacters); // deterministic: same text always gives the same vocabulary

        $characterToId = array_flip($distinctCharacters);

        return new self($distinctCharacters, $characterToId);
    }

    /**
     * Text → token IDs, e.g. "emma" → [5, 13, 13, 1].
     *
     * @return array<int, int>
     */
    public function encode(string $text): array
    {
        $tokenIds = [];
        foreach (str_split($text) as $character) {
            if (!isset($this->characterToId[$character])) {
                throw new \InvalidArgumentException(
                    "Character \"{$character}\" is not in the vocabulary — the model has no ID for it. " .
                    'This is the out-of-vocabulary problem; real tokenizers avoid it by working on raw bytes.'
                );
            }
            $tokenIds[] = $this->characterToId[$character];
        }

        return $tokenIds;
    }

    /**
     * Token IDs → text. decode(encode($text)) must always return $text unchanged.
     *
     * @param array<int, int> $tokenIds
     */
    public function decode(array $tokenIds): string
    {
        $characters = [];
        foreach ($tokenIds as $tokenId) {
            $characters[] = $this->idToCharacter[$tokenId];
        }

        return implode('', $characters);
    }

    public function vocabularySize(): int
    {
        return count($this->idToCharacter);
    }

    /** @return array<int, string> */
    public function vocabulary(): array
    {
        return $this->idToCharacter;
    }
}
