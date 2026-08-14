/**
 * The one place a sign image is looked up.
 *
 * The mapping itself comes from the backend (`GET /api/signs`), which builds
 * it from the letters the model actually knows, so swapping the artwork means
 * regenerating files - never editing a component.
 */

import { assetUrl } from "./api";
import type { SignCatalog } from "./types";

export const EMPTY_CATALOG: SignCatalog = { asset_base: "/signs", signs: {} };

/** Resolve a letter to a loadable image URL, or null if there is no sign. */
export function signImage(
  catalog: SignCatalog,
  character: string,
): string | null {
  const path = catalog.signs[character.toUpperCase()];
  return path ? assetUrl(path) : null;
}

export function supportedLetters(catalog: SignCatalog): string[] {
  return Object.keys(catalog.signs).sort();
}
