// Reading the trial brief: the model table, the prompt manifest, and the catalog built from them.
//
// The file named by MAKEFUN_PROMPT_FILE mixes prose with two JSON documents, and only the documents
// matter. Each carries a `schema` string under this competition's prefix (`free-model-aggregator-02/`);
// this module scans the text for top-level objects, parses the ones that carry such a schema, and
// picks its two documents out by the arrays they hold:
//
//   - the model table (`free-model-aggregator-02/models/v1`): one row per model the gateway can
//     route, keyed by the dialable id (`openrouter/<vendor>/<model>:free`), with the house's rating
//     in milli-points and the model's abilities (tools, vision, JSON mode, context length, free
//     flag). Only free rows are ever routed or catalogued; a paid row is dead weight here.
//   - the prompt manifest (`free-model-aggregator-02/prompts/v1`): the six prompts in the order
//     the client will send them, each with its id, its wire (`openai` or `anthropic`), whether it
//     streams, its weight (1 or 2), and the requirements a serving model must meet. The server
//     routes each arriving request by its manifest row, so the manifest is parsed before the
//     first request is answered.
//
// Either document may spell its array with the sibling competition's key (`ratings`, `manifest`),
// and a single document may carry both arrays, so a document is recognised by what it holds and
// never by a key alone. The surrounding prose is never parsed for meaning.
//
// Nothing in this module reads a clock or a random source: the same brief yields the same catalog
// on the live pass and on its replay, which is what lets `--list` and the register body agree
// byte for byte.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this entry reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * Every top-level JSON object in `text` that carries a string `schema`, in the order they appear.
 * The scan is brace-matched rather than line-based, because the brief's prose may hold braces of
 * its own; an object that does not parse, or parses without a schema, is simply passed over.
 */
function scanSchemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = closingBraceIndex(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone JSON object: keep scanning.
    }
  }
  return found;
}

/** The index of the `}` that closes the object opening at `start`, or -1; braces inside strings do not count. */
function closingBraceIndex(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** A value as a string: itself when it is one, '' for anything else. */
function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** A value as a finite number, or null when it is not one. */
function asFiniteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A value as a boolean, or null when it is not one (an unstated capability stays unknown). */
function asBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this competition's schema prefix that carries one of `keys` as an
 * array, returned with the key it was found under. Accepting several spellings is deliberate:
 * a brief written with the sibling competition's key names still reads correctly.
 */
function findArrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row, normalised. The dialable id is the row's `id` when it gives one, else its `model`
 * interpreted under the provider the row names: a row that spells the bare OpenRouter id
 * (`vendor/model:free`) and names `openrouter` is routed the way the gateway spells it
 * (`openrouter/vendor/model:free`). The brief's table is a free-model table, so a row counts as
 * free unless it explicitly says `free: false`; anything else would route a paid model.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const named = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating_milli: asFiniteNumber(v.rating_milli) ?? 0,
    context_length: asFiniteNumber(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: v.free !== false,
  };
}

/** One prompt row, normalised: only the requirements the row actually states survive. */
function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = scanSchemaDocuments(text);
  const modelsDoc = findArrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = findArrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog document the brief's model table makes: one provider, its free models, once each.
 *
 * Two details are load-bearing. The base_url is the provider's public endpoint, never the
 * gateway's trial-local address, because the gateway's origin differs between a live pass and
 * its replay while the catalog sits inside the state export a replay must reproduce byte for
 * byte. And the model set is exactly the brief's free ids — the fourteen the catalog column
 * scores — so this set is what gets registered and what both GET /v1/models answers list:
 * a missing id loses the column just as surely as an extra one.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
