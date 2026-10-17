/**
 * Embedding 客户端：走**外部 HTTP API**，不依赖本地 Ollama / ONNX / 原生模块。
 *
 * 为什么不用 NexQ 的做法：NexQ 的 `rag/embedder.rs` 是 Ollama HTTP 客户端
 * （`POST /api/embed`，模型名硬编码字符上限）。要在本机跑起来得额外装
 * ~1GB 的 Ollama 运行时 + 模型，冷启动 2–5s，且中文效果一般。
 *
 * 这里改为 OpenAI 兼容的 `/v1/embeddings`：
 *   * 建索引是一次性离线任务（11 万字 ≈ ¥0.05）；
 *   * 运行时每次提问**只调一次** query embedding（约 0.3s）；
 *   * 换 embedding 服务只需改 baseUrl + model，不碰代码。
 *
 * 兼容：阿里云百炼 `/compatible-mode/v1`、硅基流动、OpenAI、任何兼容端点。
 * 网络抖动做指数退避重试；批量失败时自动降半批重试（对齐 NexQ 的 400 降半策略）。
 */

export interface EmbeddingSettings {
  providerId: string;
  baseUrl: string;
  model: string;
  dim?: number;
  batchSize?: number;
}

export interface EmbeddingResult {
  vectors: number[][];
  /** 实际维度，用于校验索引是否过期 */
  dim: number;
  /** 本次调用耗时（毫秒） */
  latencyMs: number;
  /** 服务端报告的用量（部分厂商返回） */
  usageTokens?: number;
}

// 预设与默认值的**单一事实源在 shared/ragDefaults.ts**（主进程与渲染进程共用一份）。
// 这里只做转发，避免界面显示的预设与后端实际请求的端点漂移。
export {
  EMBEDDING_PRESETS,
  DEFAULT_EMBEDDING_PRESET,
  DEFAULT_MIN_SCORE,
  findEmbeddingPreset,
} from '../../shared/ragDefaults';
export type { EmbeddingProviderPreset } from '../../shared/ragDefaults';

const DEFAULT_BATCH = 16;
const MAX_RETRY = 3;

/** 把 baseUrl 规范化到 `/embeddings` 端点，容错用户少写/多写 `/v1`。 */
export function embeddingsEndpoint(baseUrl: string): string {
  let b = baseUrl.trim().replace(/\/+$/, '');
  if (!b) throw new Error('embedding baseUrl 为空');
  if (/\/embeddings$/.test(b)) return b;
  // 剥掉已知的同类子路径：用户从厂商文档复制一个 chat/models/audio 端点过来
  // 是很常见的，不做这步会拼出 `…/v1/chat/completions/v1/embeddings`
  // 这种 404 且报错完全看不懂的 URL。
  b = b.replace(
    /\/(?:chat\/completions|completions|responses|models|audio\/[a-z-]+|rerank|images\/generations)$/i,
    '',
  );
  if (/\/compatible-mode\/v1$/.test(b) || /\/v1$/.test(b)) return `${b}/embeddings`;
  // 用户只填到域名：补 /v1/embeddings
  return `${b}/v1/embeddings`;
}

interface OpenAiEmbeddingResponse {
  data?: { embedding?: number[]; index?: number }[];
  usage?: { total_tokens?: number };
  error?: { message?: string; code?: string };
}

function errMessage(status: number, body: string): string {
  try {
    const j = JSON.parse(body) as OpenAiEmbeddingResponse;
    if (j.error?.message) return `HTTP ${status}: ${j.error.message}`;
  } catch {
    /* 非 JSON 错误体 */
  }
  return `HTTP ${status}: ${body.slice(0, 300)}`;
}

async function postOnce(
  endpoint: string,
  apiKey: string,
  model: string,
  input: string[],
  dim: number | undefined,
  signal?: AbortSignal,
): Promise<OpenAiEmbeddingResponse> {
  /** 部分厂商/模型不支持 dimensions 参数，会回 400 —— 这时要能自动降级。 */
  const build = (withDim: boolean): Record<string, unknown> => {
    const body: Record<string, unknown> = { model, input, encoding_format: 'float' };
    if (withDim && dim && dim > 0) body.dimensions = dim;
    return body;
  };

  const send = async (withDim: boolean) => {
    // 已经取消时不要发请求，直接把原生取消错误交给上层。
    // 这样调用方看到的是标准 AbortError，且一次网络请求都不会发出。
    if (signal?.aborted) {
      throw signal.reason instanceof Error ? signal.reason : new AbortedError();
    }
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(build(withDim)),
      signal,
    });
    const text = await res.text();
    return { res, text };
  };

  let { res, text } = await send(!!dim);
  // dimensions 被拒 → 去掉该参数重试一次，用服务端默认维度
  if (!res.ok && res.status === 400 && dim) {
    const again = await send(false);
    res = again.res;
    text = again.text;
  }
  if (!res.ok) throw new Error(errMessage(res.status, text));
  try {
    return JSON.parse(text) as OpenAiEmbeddingResponse;
  } catch {
    throw new Error(`embedding 返回不是 JSON：${text.slice(0, 200)}`);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * 返回条数与请求不符。
 * 这类错误**不能**被"降半批重试"吞掉：重试同样会不符，最终会静默产出
 * 含 undefined 占位的向量数组，让索引里混入错位/空向量——那是一类不会
 * 立刻报错、却让检索悄悄失效的故障。所以单独标记并立即抛出。
 */
class ResponseShapeError extends Error {}

/**
 * 被主动取消。
 * 取消后**绝不允许**继续重试或降半批——实测不处理这一点时，
 * 取消一个 16 条的批次会跑完 15 次请求、耗时 4.5 秒才停，
 * 面试现场表现为"点了停止但界面还在转、还在花钱"。
 *
 * 刻意保留原生 AbortError 的形态（name === 'AbortError'）：调用方通常按
 * `e.name === 'AbortError'` 区分"用户取消"与"真失败"，换成自定义类会让它们失配。
 */
class AbortedError extends Error {
  constructor(cause?: unknown) {
    super('embedding 请求已取消');
    this.name = 'AbortError';
    if (cause !== undefined) (this as { cause?: unknown }).cause = cause;
  }
}

/** 统一的取消判定：原生 AbortError 与"signal 已 aborted"都要认。 */
function isAbort(e: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  const name = (e as { name?: string } | undefined)?.name;
  return name === 'AbortError' || name === 'TimeoutError';
}

/**
 * 批量向量化。自动分批、重试、**降半批重试**。
 * 返回顺序与输入严格一致（按 index 排序，不依赖服务端顺序）。
 */
export async function embedTexts(
  settings: EmbeddingSettings,
  apiKey: string,
  texts: string[],
  opts: { signal?: AbortSignal; onProgress?: (done: number, total: number) => void } = {},
): Promise<EmbeddingResult> {
  if (texts.length === 0) return { vectors: [], dim: 0, latencyMs: 0 };
  if (!apiKey) throw new Error('未配置 embedding API Key');
  const endpoint = embeddingsEndpoint(settings.baseUrl);
  const batchSize = Math.max(1, settings.batchSize ?? DEFAULT_BATCH);
  const started = Date.now();
  const vectors: number[][] = new Array(texts.length);
  let usageTokens = 0;
  let dim = 0;

  for (let start = 0; start < texts.length; start += batchSize) {
    if (opts.signal?.aborted) throw new AbortedError();
    let size = Math.min(batchSize, texts.length - start);
    let done = false;
    // 降半批重试：批太大被服务端拒绝时，逐步减半
    while (!done) {
      const slice = texts.slice(start, start + size);
      let attempt = 0;
      for (;;) {
        if (opts.signal?.aborted) throw new AbortedError();
        try {
          const r = await postOnce(endpoint, apiKey, settings.model, slice, settings.dim, opts.signal);
          const items = r.data ?? [];
          if (items.length !== slice.length) {
            // 形状错误不可重试（见 ResponseShapeError 注释）
            throw new ResponseShapeError(`返回条数不符：期望 ${slice.length}，实际 ${items.length}`);
          }
          // 按 index 归位：**必须用服务端返回的 index**，不能用数组下标——
          // 服务端可以乱序返回，若按数组位置落盘，向量就会和错的文本绑定，
          // 检索会静默失效（不报错但答非所问）。
          const ordered = items
            .map((it, i) => ({ i: it.index ?? i, v: it.embedding }))
            .sort((a, b) => a.i - b.i);
          const seenIdx = new Set<number>();
          for (const it of ordered) {
            if (!it.v || it.v.length === 0) throw new ResponseShapeError('返回了空向量');
            if (it.i < 0 || it.i >= slice.length) {
              throw new ResponseShapeError(`返回的 index 越界：${it.i}（本批 ${slice.length} 条）`);
            }
            if (seenIdx.has(it.i)) throw new ResponseShapeError(`返回的 index 重复：${it.i}`);
            seenIdx.add(it.i);
            vectors[start + it.i] = it.v;
            dim = it.v.length;
          }
          usageTokens += r.usage?.total_tokens ?? 0;
          done = true;
          break;
        } catch (e) {
          // 取消：立即停止，不重试、不降半批（见 AbortedError 注释）。
          // 抛原始错误，让调用方拿到标准 AbortError 形态。
          if (isAbort(e, opts.signal)) throw e;
          if (e instanceof ResponseShapeError) throw e;
          attempt++;
          const msg = (e as Error).message;
          // 认证/参数类错误重试无意义，直接抛
          if (/HTTP 40[13]/.test(msg) && !/429/.test(msg)) throw e;
          if (attempt >= MAX_RETRY) {
            if (size > 1) {
              // 降半批：可能是单条文本过长或批太大
              size = Math.max(1, Math.floor(size / 2));
              attempt = 0;
              break; // 跳出重试循环，重跑 while
            }
            throw e;
          }
          await sleep(300 * attempt);
        }
      }
    }
    opts.onProgress?.(Math.min(start + size, texts.length), texts.length);
  }

  // 纵深防御：即使上游校验全通过，也确认没有留下洞。
  // 带洞的向量数组会被 indexer 原样入库，那些片段永远检索不到，
  // 而且**不报任何错**——属于最隐蔽的一类故障。
  const hole = vectors.findIndex((v) => !v || v.length === 0);
  if (hole >= 0) throw new ResponseShapeError(`第 ${hole} 条向量缺失（索引未完整生成）`);

  return { vectors, dim, latencyMs: Date.now() - started, usageTokens };
}

/** 单条查询向量化（运行时热路径，每次提问只调一次）。 */
export async function embedQuery(
  settings: EmbeddingSettings,
  apiKey: string,
  query: string,
  signal?: AbortSignal,
): Promise<number[]> {
  const r = await embedTexts(settings, apiKey, [query], { signal });
  return r.vectors[0] ?? [];
}

/** 连通性测试：返回可读结果供设置面板显示。 */
export async function testEmbedding(
  settings: EmbeddingSettings,
  apiKey: string,
  signal?: AbortSignal,
): Promise<{ ok: boolean; messageZh: string; dim?: number; latencyMs?: number }> {
  try {
    const t0 = Date.now();
    const r = await embedTexts({ ...settings, batchSize: 1 }, apiKey, ['连通性测试'], { signal });
    const dim = r.dim;
    return {
      ok: true,
      dim,
      latencyMs: Date.now() - t0,
      messageZh: `连接成功：${settings.model}，维度 ${dim}，${Date.now() - t0}ms`,
    };
  } catch (e) {
    return { ok: false, messageZh: `连接失败：${(e as Error).message}` };
  }
}
