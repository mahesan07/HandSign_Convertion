/**
 * Text -> Sign.
 *
 * Typing is debounced and every in-flight translation is cancelled when the
 * next keystroke arrives, so the strip keeps up with fast typing without
 * queueing stale requests.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { SignStrip, SignTile } from "../components/SignTile";
import { Button, Card, Notice } from "../components/ui";
import { api, ApiError } from "../lib/api";
import type { SignCatalog, SignTranslation } from "../lib/types";

const DEBOUNCE_MS = 180;

const EXAMPLES = [
  "HELLO HOW ARE YOU",
  "THANK YOU",
  "I NEED HELP PLEASE",
];

interface Props {
  catalog: SignCatalog;
}

export function TextToSignView({ catalog }: Props) {
  const [text, setText] = useState("HELLO");
  const [translation, setTranslation] = useState<SignTranslation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | undefined>();
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!text.trim()) {
      setTranslation(null);
      setError(null);
      return;
    }

    const timer = window.setTimeout(async () => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const result = await api.translateToSign(text, controller.signal);
        setTranslation(result);
        setError(null);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Could not translate that text.",
        );
      }
    }, DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [text]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const alphabet = Object.keys(catalog.signs).sort();

  const handleSelect = useCallback(
    (index: number) => setActiveIndex((current) => (current === index ? undefined : index)),
    [],
  );

  return (
    <div className="stack">
      <Card title="Text to translate">
        <div className="stack">
          <label htmlFor="text-to-sign" className="sr-only">
            Text to show as hand signs
          </label>
          <textarea
            id="text-to-sign"
            className="composer"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setActiveIndex(undefined);
            }}
            placeholder="Type a message, for example: HELLO HOW ARE YOU"
            spellCheck={false}
            maxLength={500}
          />
          <div className="controls">
            {EXAMPLES.map((example) => (
              <Button key={example} onClick={() => setText(example)}>
                {example}
              </Button>
            ))}
            <Button variant="danger" onClick={() => setText("")} disabled={!text}>
              Clear
            </Button>
          </div>
        </div>
      </Card>

      {error && <Notice tone="danger" title="Translation failed" detail={error} />}

      <Card
        title="Hand signs"
        action={
          translation && (
            <span className="meta-row">
              {translation.sign_count} sign
              {translation.sign_count === 1 ? "" : "s"}
            </span>
          )
        }
      >
        {translation && translation.words.length > 0 ? (
          <>
            <div style={{ overflowX: "auto" }}>
              <SignStrip
                words={translation.words}
                activeIndex={activeIndex}
                onSelect={handleSelect}
              />
            </div>
            {translation.unsupported.length > 0 && (
              <Notice
                tone="warn"
                title="Some characters have no sign"
                detail={`No hand sign is available for: ${translation.unsupported.join(", ")}. They are shown as plain characters.`}
              />
            )}
          </>
        ) : (
          <p className="chips__empty">
            Type something above and the matching hand signs appear here.
          </p>
        )}
      </Card>

      <Card title={`Alphabet reference (${alphabet.length} signs)`}>
        <div className="alphabet-grid">
          {alphabet.map((letter) => (
            <SignTile
              key={letter}
              token={{
                character: letter,
                kind: "letter",
                asset: catalog.signs[letter],
                label: `Hand sign for the letter ${letter}`,
              }}
            />
          ))}
        </div>
        <p className="chips__empty" style={{ marginTop: "0.75rem" }}>
          Each illustration is the median hand pose recorded for that letter in
          this project's own training data, drawn as a skeleton and shown from
          the signer's point of view.
        </p>
      </Card>
    </div>
  );
}
