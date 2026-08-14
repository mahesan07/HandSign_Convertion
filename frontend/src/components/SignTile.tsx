/** One hand-sign tile, and the strip of words built from them. */

import { useState } from "react";
import { assetUrl } from "../lib/api";
import type { SignToken, SignWord } from "../lib/types";

export function SignTile({
  token,
  active = false,
  onSelect,
}: {
  token: SignToken;
  active?: boolean;
  onSelect?: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const showImage = token.asset && !failed;

  const className = [
    "sign-tile",
    onSelect ? "sign-tile--interactive" : "",
    active ? "sign-tile--active" : "",
    !showImage ? "sign-tile--missing" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const content = (
    <>
      {showImage ? (
        <img
          className="sign-tile__image"
          src={assetUrl(token.asset ?? "")}
          alt=""
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="sign-tile__glyph" aria-hidden="true">
          {token.kind === "letter" ? "?" : token.character}
        </span>
      )}
      <span className="sign-tile__caption">{token.character}</span>
    </>
  );

  if (onSelect) {
    return (
      <button
        type="button"
        className={className}
        onClick={onSelect}
        aria-pressed={active}
        aria-label={token.label}
        title={token.label}
      >
        {content}
      </button>
    );
  }

  return (
    <span className={className} role="img" aria-label={token.label} title={token.label}>
      {content}
    </span>
  );
}

export function SignStrip({
  words,
  activeIndex,
  onSelect,
}: {
  words: SignWord[];
  activeIndex?: number;
  onSelect?: (index: number) => void;
}) {
  // Running offset so each tile has a stable index across the whole sentence,
  // which is what "currently selected sign" refers to.
  const offsets: number[] = [];
  words.reduce((total, word, index) => {
    offsets[index] = total;
    return total + word.signs.length;
  }, 0);

  return (
    <div className="sign-strip">
      {words.map((word, wordIndex) => (
        <div className="sign-word" key={`${word.text}-${wordIndex}`}>
          <span className="sign-word__label">{word.text}</span>
          <div className="sign-word__signs">
            {word.signs.map((token, signIndex) => {
              const index = offsets[wordIndex] + signIndex;
              return (
                <SignTile
                  key={`${token.character}-${signIndex}`}
                  token={token}
                  active={activeIndex === index}
                  onSelect={onSelect ? () => onSelect(index) : undefined}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
