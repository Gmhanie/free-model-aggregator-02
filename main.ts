// The best live free-model aggregator: one server, two client wires, and a real OpenRouter free
// model behind every answer, dialled through make.fun's gateway.
//
// STRATEGY (each choice is a scoring column)
//   quality (60%): every prompt is routed to the highest-rated free model in the brief that meets
//     the prompt's requirements (the brief's manifest row for the arriving request, unioned with
//     what the body itself declares). Ratings come from the brief, so the pick is a pure function
//     of the brief and the request.
//   failover (10%): a refused call is never the end of a prompt while another candidate stands.
//     A 429 moves straight to the next candidate (a rate limit does not clear in seconds); a 5xx
//     earns its model one retry, then the next candidate; a 4xx moves on (the request shape is
//     that model's problem, not the next one's); a 401 stops the prompt at once (no key was
//     pushed, so every model would refuse identically). Every refusal a prompt recovered from is
//     reported on the answer in the x-makefun-upstream-failures header — the only way the
//     failover column can see a recovered refusal. Calls that met an upstream error do not count
//     toward the tie-break, so failing over is free there; the caps below exist for the client's
//     30-second patience and the trial's 60-call ceiling, not for the tie-break.
//   cooldowns: a model that refuses sits out the next prompt, then returns to its rating place.
//     One prompt is long enough for a per-minute limit to clear; if the refusal was a daily cap,
//     the model refuses once more, cheaply, and the prompt fails over exactly as it would have —
//     while a model struck for the whole trial would tax every later prompt with the rating gap.
//   served (15%) / wire (10%): the client body passes to the gateway whole (system prompt, tools,
//     JSON mode, temperature), with two guarantees added: a model id from the brief, and an
//     output cap on every call — the gateway refuses an uncapped request, so a body that names
//     neither max_tokens nor max_completion_tokens gets a default cap. Answers come back in the
//     client's own wire, streamed or not as asked, with the dialled id on every answer and chunk.
//   catalog (5%): exactly the brief's free ids — in the registered catalog and in both
//     GET /v1/models answers, once each, no more and no fewer.
//
// DETERMINISM
//   Every choice derives from the brief, the requests, and the gateway's answers seen so far
//   (cooldowns are a function of past refusals, never of the wall clock), so a replay that serves
//   the recorded answers back walks the same route and leaves the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway dials one prompt may cost, retries included (the trial allows 60 in all). */
const MAX_DIALS_PER_PROMPT = 6;
/** How long one non-streaming dial may take before it counts as a refusal and the route moves on. */
const DIAL_TIMEOUT_MS = 15_000;
/** How long one streaming dial may run; the client waits 30 seconds for the whole answer. */
const STREAM_TIMEOUT_MS = 28_000;
/** The output cap a dial gets when the client's body names none: the gateway refuses an uncapped call. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

/** The Authorization header value of a gateway dial: the vendor's scheme over the trial's dummy key. */
function gatewayAuth(): string {
  return ['Bearer', GATEWAY_KEY].join(' ');
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Refusals per model this trial: the ordering state among models that have refused. */
const refusals = new Map<string, number>();
/**
 * The arrival index from which a refused model is dialled first-choice again. A refusal at
 * arrival i sets it to i + 2: the model sits out exactly one prompt, then returns to its rating
 * place. A per-minute limit has cleared by then; a daily cap simply refuses again, cheaply,
 * and the prompt fails over as it would have anyway. Arrival counts and past answers are all
 * this reads — never the wall clock — so a replay walks the same route.
 */
const eligibleFrom = new Map<string, number>();
/** Each free model's place in the brief, the final tie-break of the route order. */
const briefOrder = new Map<string, number>(freeModels(brief).map((m, i) => [m.id, i]));

/** One refused gateway call, as the failover trail reports it. */
interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The candidates for one prompt, in the order this entry dials them: models that meet the
 * requirements and are not sitting out a cooldown, by the brief's rating, highest first, then
 * by brief order — then, as a fallback that keeps a prompt servable, the cooling models in the
 * same order. A cooling model is dialled only when no eligible model meets the prompt.
 */
function candidatesFor(requires: Requirement, arrival: number): ModelRow[] {
  const byRank = (a: ModelRow, b: ModelRow): number =>
    b.rating_milli - a.rating_milli ||
    (refusals.get(a.id) ?? 0) - (refusals.get(b.id) ?? 0) ||
    (briefOrder.get(a.id) ?? 0) - (briefOrder.get(b.id) ?? 0);
  const meeting = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const eligible = meeting.filter((model) => (eligibleFrom.get(model.id) ?? 0) <= arrival).sort(byRank);
  const cooling = meeting.filter((model) => (eligibleFrom.get(model.id) ?? 0) > arrival).sort(byRank);
  return [...eligible, ...cooling];
}

/** Strike a model for a refusal at this arrival: it sits out the next prompt, then returns. */
function noteRefusal(model: string, arrival: number): void {
  refusals.set(model, (refusals.get(model) ?? 0) + 1);
  eligibleFrom.set(model, arrival + 2);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, unioned with the body's. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    tools: entry.requires.tools === true || fromBody.tools === true,
    vision: entry.requires.vision === true || fromBody.vision === true,
    json: entry.requires.json === true || fromBody.json === true,
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** Requirements read off a request body: declared tools, an image in any message, a JSON response format. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  headers: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), headers);
  else sendJson(res, status, openAIError(message, type, null), headers);
}

/** The failover trail as the header the column reads: at most 8 rows, in the order the calls refused. */
function trailHeader(trail: readonly Refusal[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, 8)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A dial body with an output cap guaranteed: the gateway refuses a request that names none. */
function withCap(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_MAX_TOKENS };
}

/** The refusal code a gateway error body carries, or a plain name for its status. */
function refusalCode(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code ?? parsed.error.type;
      if (typeof code === 'string' && code !== '') return code;
      if (typeof code === 'number') return String(code);
    }
  } catch {
    // Not JSON: the status names it.
  }
  return `http_${status}`;
}

/** The outcome of dialling one prompt through the candidates: a live response, or the last refusal. */
interface Served {
  upstream: Response;
  model: string;
  trail: Refusal[];
}
interface GaveUp {
  upstream: null;
  trail: Refusal[];
  lastStatus: number;
  lastText: string;
}

/**
 * Dial one prompt's gateway body (built per model by `build`) through the candidates in route
 * order until one answers. The refusal policy, per model: a 429 or a 4xx moves to the next
 * candidate at once; a 5xx or an unreachable gateway earns the same model one retry; a 401 ends
 * the prompt (no key was pushed — every candidate would refuse identically). Every refusal is
 * struck against its model and appended to the trail the answer reports.
 */
async function dialWithFailover(requires: Requirement, build: (model: string) => Json, streaming: boolean, arrival: number): Promise<Served | GaveUp> {
  const trail: Refusal[] = [];
  let lastStatus = 502;
  let lastText = 'no free model in the brief meets this request';
  let dials = 0;
  for (const candidate of candidatesFor(requires, arrival)) {
    let retried = false;
    while (dials < MAX_DIALS_PER_PROMPT) {
      dials += 1;
      let upstream: Response;
      try {
        upstream = await fetch(chatCompletionsUrl(), {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: gatewayAuth(),
            ...(streaming ? { accept: 'text/event-stream' } : {}),
          },
          body: JSON.stringify(withCap(build(candidate.id))),
          signal: AbortSignal.timeout(streaming ? STREAM_TIMEOUT_MS : DIAL_TIMEOUT_MS),
        });
      } catch (err) {
        trail.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
        noteRefusal(candidate.id, arrival);
        lastStatus = 502;
        lastText = (err as Error).message;
        if (!retried) {
          retried = true;
          continue;
        }
        break;
      }
      if (upstream.ok && upstream.body !== null) return { upstream, model: candidate.id, trail };
      const text = await upstream.text().catch(() => '');
      const status = upstream.status;
      trail.push({ status, code: refusalCode(text, status), model: candidate.id });
      noteRefusal(candidate.id, arrival);
      lastStatus = status;
      lastText = text;
      if (status === 401) return { upstream: null, trail, lastStatus, lastText };
      if (status >= 500 && !retried) {
        retried = true;
        continue;
      }
      break;
    }
    if (dials >= MAX_DIALS_PER_PROMPT) break;
  }
  return { upstream: null, trail, lastStatus, lastText };
}

/** A prompt no candidate would serve, answered as a 502 in the client's wire, trail attached. */
function gaveUp(res: ServerResponse, wire: 'openai' | 'anthropic', outcome: GaveUp): void {
  const message = `the gateway answered HTTP ${outcome.lastStatus}${outcome.lastText === '' ? '' : `: ${outcome.lastText.slice(0, 500)}`}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', message, trailHeader(outcome.trail));
  else sendError(res, wire, 502, 'upstream_error', message, trailHeader(outcome.trail));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const streaming = body.stream === true;
  const arrival = manifestIndex;
  const outcome = await dialWithFailover(
    requirementsFor(body),
    (model) => (streaming ? { ...body, model, stream_options: { include_usage: true } } : { ...body, model }),
    streaming,
    arrival,
  );
  if (outcome.upstream === null) return gaveUp(res, 'openai', outcome);
  const { upstream, model, trail } = outcome;
  if (streaming) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trailHeader(trail) });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trailHeader(trail));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const streaming = body.stream === true;
  const arrival = manifestIndex;
  const outcome = await dialWithFailover(requirementsFor(body), (model) => anthropicToOpenAI(body, model), streaming, arrival);
  if (outcome.upstream === null) return gaveUp(res, 'anthropic', outcome);
  const { upstream, model, trail } = outcome;
  if (streaming) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trailHeader(trail) });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trailHeader(trail));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
