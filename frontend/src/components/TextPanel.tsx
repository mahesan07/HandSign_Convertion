/** The sentence being built, and the controls that edit it. */

import type { TextState } from "../lib/types";
import { Button } from "./ui";

interface Props {
  text: TextState;
  onSpace: () => void;
  onBackspace: () => void;
  onDeleteWord: () => void;
  onClear: () => void;
  disabled?: boolean;
}

export function TextPanel({
  text,
  onSpace,
  onBackspace,
  onDeleteWord,
  onClear,
  disabled = false,
}: Props) {
  const empty = !text.text;

  return (
    <div className="stack">
      <div className="sentence sunken">
        {empty ? (
          <span className="sentence__empty">
            Your sentence appears here as you sign. Hold each sign steady until
            the bar fills.
          </span>
        ) : (
          <>
            {text.words.length > 0 && <span>{text.words.join(" ")} </span>}
            {text.current_word && (
              <span className="sentence__current">{text.current_word}</span>
            )}
            <span className="caret" aria-hidden="true" />
          </>
        )}
      </div>

      {/* Announce only completed text, so screen readers are not spammed. */}
      <p className="sr-only" aria-live="polite">
        {empty ? "No text yet." : `Current text: ${text.text}`}
      </p>

      <div className="meta-row">
        <span>
          Current word:{" "}
          <strong style={{ color: "var(--text)" }}>
            {text.current_word || "-"}
          </strong>
        </span>
        <span>
          {text.words.length} word{text.words.length === 1 ? "" : "s"} ·{" "}
          {text.text.replace(/\s/g, "").length} letters
        </span>
      </div>

      <div className="controls">
        <Button onClick={onSpace} disabled={disabled} hint="Space">
          Finish word
        </Button>
        <Button onClick={onBackspace} disabled={disabled || empty} hint="Bksp">
          Delete letter
        </Button>
        <Button onClick={onDeleteWord} disabled={disabled || empty}>
          Delete word
        </Button>
        <Button
          variant="danger"
          onClick={onClear}
          disabled={disabled || empty}
          hint="Esc"
        >
          Clear all
        </Button>
      </div>
    </div>
  );
}
