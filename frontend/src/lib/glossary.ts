// Shared glossary. examples/glossary.json is the single
// source, loaded identically by the backend (backend/app/explain/glossary.py).

import glossaryJson from "../../../examples/glossary.json";

export type ReadingLevel = "curious" | "student" | "expert";

export interface GlossaryAnalogy {
  text: string;
  breaks_when: string;
}

export interface GlossaryTerm {
  display: string;
  curious: string;
  student: string;
  expert: string;
  analogy?: GlossaryAnalogy;
  see_also?: string[];
}

export interface StandaloneAnalogy {
  id: string;
  term_id: string;
  title: string;
  text: string;
  breaks_when: string;
}

interface GlossaryFile {
  terms: Record<string, GlossaryTerm>;
  analogies: StandaloneAnalogy[];
}

const data = glossaryJson as unknown as GlossaryFile;

export const GLOSSARY: Record<string, GlossaryTerm> = data.terms;
export const ANALOGIES: StandaloneAnalogy[] = data.analogies;

export function termDefinition(id: string, level: ReadingLevel): string | null {
  const t = GLOSSARY[id];
  return t ? t[level] : null;
}

/** Parse claim text into plain strings and glossary markers. */
export interface TextPart {
  kind: "text" | "term";
  value: string;
  termId?: string;
  display?: string;
}

const MARKER_RE = /\{\{term:([a-z_]+)\|([^{}]*)\}\}/g;

export function parseTermMarkers(text: string): TextPart[] {
  const parts: TextPart[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER_RE)) {
    const idx = m.index ?? 0;
    if (idx > last) parts.push({ kind: "text", value: text.slice(last, idx) });
    parts.push({ kind: "term", value: m[0], termId: m[1], display: m[2] });
    last = idx + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", value: text.slice(last) });
  return parts;
}
