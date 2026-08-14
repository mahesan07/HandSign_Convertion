/** Word and sentence suggestions, with an honest label on where they came from. */

import type { Suggestions } from "../lib/types";
import { Badge, Card } from "./ui";

const SOURCE_LABEL: Record<Suggestions["source"], string> = {
  local: "Instant",
  gemini: "Smart",
  cache: "Smart (cached)",
};

interface Props {
  suggestions: Suggestions;
  onPickWord: (word: string) => void;
  onPickSentence: (sentence: string) => void;
}

export function SuggestionPanel({
  suggestions,
  onPickWord,
  onPickSentence,
}: Props) {
  const { word_suggestions, sentence_suggestions, source, llm_pending, notice } =
    suggestions;

  return (
    <div className="stack">
      <Card
        title="Word suggestions"
        action={
          <span
            style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}
          >
            {llm_pending && (
              <>
                <span className="spinner" aria-hidden="true" />
                <span className="sr-only">Fetching smarter suggestions</span>
              </>
            )}
            <Badge muted={source === "local"}>{SOURCE_LABEL[source]}</Badge>
          </span>
        }
      >
        {word_suggestions.length ? (
          <div className="chips">
            {word_suggestions.map((word) => (
              <button
                key={word}
                type="button"
                className="chip"
                onClick={() => onPickWord(word)}
                aria-label={`Use the word ${word}`}
              >
                {word}
              </button>
            ))}
          </div>
        ) : (
          <p className="chips__empty">
            Start spelling and suggested words will appear here.
          </p>
        )}
      </Card>

      <Card title="Sentence suggestions">
        {sentence_suggestions.length ? (
          <div className="chips chips--stacked">
            {sentence_suggestions.map((sentence) => (
              <button
                key={sentence}
                type="button"
                className="chip chip--sentence"
                onClick={() => onPickSentence(sentence)}
                aria-label={`Replace my text with: ${sentence}`}
              >
                {sentence}
              </button>
            ))}
          </div>
        ) : (
          <p className="chips__empty">
            Complete a word to see full sentences you might mean.
          </p>
        )}
      </Card>

      {notice && (
        <p className="chips__empty" role="status">
          {notice} Instant suggestions are still working.
        </p>
      )}
    </div>
  );
}
