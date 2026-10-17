/**
 * 检索：向量余弦 + 中文字符 bigram 关键词加分 + 邻接扩展 + **低置信判定**。
 *
 * 与 NexQ 的三点关键差异（都是为了"防瞎编"）：
 *
 * 1. **分数是原始余弦，不是 RRF 融合分**。
 *    NexQ `search.rs` 用 RRF `w/(60+rank)`，top-5 归一化后全在 0.95 以上，
 *    配置里的 `similarity_threshold=0.3` 永远筛不掉任何东西——"没找到资料"
 *    这个信号在它的实现里根本不存在。而"没找到"恰恰是我们最需要的负信号。
 *
 * 2. **关键词分支真的能用**。NexQ 的 FTS5 表没指定分词器，中文整段是一个
 *    token，实测中文查询 0 命中。这里用字符 bigram 覆盖度（中文无需分词器），
 *    并对专有名词/数字指标（"91.3%"、"FTS5"、"RRF"）给精确命中强加分。
 *
 * 3. **邻接扩展**。片段被切散会丢掉上下文（STAR 的"结果"常在下一段），
 *    命中后自动带上同源的相邻片段，再按预算截断。
 */
import type { FactLevel, KbSlot, RagChunk, RagEvidence, RagRetrieval } from '../../shared/protocol';

/** 关键词加分的权重上限（语义为主、关键词为辅）。 */
const KEYWORD_WEIGHT = 0.15;
/** 专有名词/指标精确命中的额外加分。 */
const EXACT_BONUS = 0.12;
/** 低于此余弦视为"资料里没找到"，直接走兜底话术。可按语料校准。 */
export const DEFAULT_MIN_SCORE = 0.45;

/** 余弦相似度。向量不匹配或零向量返回 0。 */
export function cosine(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  const s = dot / (Math.sqrt(na) * Math.sqrt(nb));
  // 浮点误差可能略微越界
  return s > 1 ? 1 : s < -1 ? -1 : s;
}

/**
 * 提取检索用的词元：
 *   * 连续中文/日韩字符 → 相邻二字 bigram（"知识服务平台" → 知识/识服/服务/务平/平台）
 *   * 拉丁词/数字 → 原样小写（"FTS5"、"91.3%"、"RRF"）
 * 这个做法对中文无需任何分词器，也不引入原生依赖。
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  const lower = text.toLowerCase();
  // 中文段落取 bigram
  const cjkRuns = lower.match(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+/g) ?? [];
  for (const run of cjkRuns) {
    if (run.length === 1) out.push(run);
    for (let i = 0; i + 1 < run.length; i++) out.push(run.slice(i, i + 2));
  }
  // 拉丁/数字词
  const words = lower.match(/[a-z0-9][a-z0-9._%+#-]*/g) ?? [];
  for (const w of words) if (w.length > 1) out.push(w);
  return out;
}

/**
 * 关键词覆盖度 0..1：查询词元里有多少比例出现在片段中。
 * 长词元命中权重更高（"91.3%" 比 "平台" 更有判别力）。
 */
export function keywordScore(queryTokens: string[], chunkText: string): number {
  if (queryTokens.length === 0) return 0;
  const hay = chunkText.toLowerCase();
  let hit = 0;
  let total = 0;
  for (const t of queryTokens) {
    const w = t.length >= 4 ? 2 : 1;
    total += w;
    if (hay.includes(t)) hit += w;
  }
  return total === 0 ? 0 : hit / total;
}

/** 查询里的专有名词/数字指标是否被片段精确包含（强信号）。 */
export function exactHits(query: string, chunkText: string): number {
  const pats = query.match(/[A-Za-z][A-Za-z0-9+#.-]{1,}|\d+(?:\.\d+)?\s*[%％倍条例次万kK]?/g) ?? [];
  const hay = chunkText.toLowerCase();
  let n = 0;
  for (const p of pats) {
    // 单个字母/纯数字太弱，忽略
    if (p.length < 2) continue;
    if (hay.includes(p.toLowerCase())) n++;
  }
  return n;
}

export interface SearchOptions {
  topK?: number;
  minScore?: number;
  neighborExpand?: boolean;
  /** 只在这些槽位里检索；缺省=全部 */
  slots?: KbSlot[];
  /**
   * 自测/调试模式：即使低置信也保留证据，便于用户看到真实分数分布。
   * 正式检索路径**不能**开——契约是"低置信 ⇒ 没有证据"。
   */
  debug?: boolean;
}

/**
 * 主检索入口。
 *
 * @param query 面试官的问题（或问题+转写片段）
 * @param queryVector 已算好的查询向量（调用方负责调 embedding，便于复用）
 */
export function search(
  store: { all(): readonly RagChunk[] },
  query: string,
  queryVector: readonly number[],
  opts: SearchOptions = {},
): RagRetrieval {
  const t0 = Date.now();
  const topK = Math.max(1, opts.topK ?? 5);
  const minScore = opts.minScore ?? DEFAULT_MIN_SCORE;
  const chunks = store.all();
  if (chunks.length === 0 || queryVector.length === 0) {
    return { evidence: [], topScore: 0, lowConfidence: true, latencyMs: Date.now() - t0 };
  }

  const q = query.trim();
  const qt = tokenize(q);
  const slotFilter = opts.slots && opts.slots.length > 0 ? new Set(opts.slots) : null;

  interface Scored {
    chunk: RagChunk;
    cos: number;
    kw: number;
    exact: number;
    total: number;
  }
  const scored: Scored[] = [];
  for (const c of chunks) {
    if (slotFilter && !slotFilter.has(c.slot)) continue;
    if (!c.vector || c.vector.length === 0) continue;
    // ★维度不符的片段直接跳过★
    // 否则它的余弦恒为 0，却仍可能靠关键词加分抢到第 1 名：
    // 实测过 [维度不符 → score 0 + bonus 0.39] 排在 [正确 → score 0.3] 前面，
    // 把一条完全无关的片段当成头号证据注入提示词。
    if (c.vector.length !== queryVector.length) continue;
    const cos = cosine(queryVector, c.vector);
    const kw = keywordScore(qt, c.text);
    const exact = exactHits(q, c.text);
    // 语义为主；关键词与专有名词只做小幅加分，避免"关键词命中但语义无关"被抬上来
    const total = cos + KEYWORD_WEIGHT * kw + EXACT_BONUS * Math.min(exact, 2);
    scored.push({ chunk: c, cos, kw, exact, total });
  }
  if (scored.length === 0) {
    return { evidence: [], topScore: 0, lowConfidence: true, latencyMs: Date.now() - t0 };
  }
  scored.sort((a, b) => b.total - a.total);

  const top = scored.slice(0, topK);
  const topCos = top[0]?.cos ?? 0;

  // 邻接扩展：把命中片段的同源前后段补进来（去重），保住被切散的上下文
  let picked: Scored[] = top;
  if (opts.neighborExpand) {
    const seen = new Set(top.map((s) => s.chunk.id));
    const extra: Scored[] = [];
    for (const s of top) {
      for (const d of [-1, 1]) {
        const c = chunks.find(
          (x) => x.slot === s.chunk.slot && x.source === s.chunk.source && x.index === s.chunk.index + d,
        );
        if (c && !seen.has(c.id)) {
          seen.add(c.id);
          extra.push({ chunk: c, cos: s.cos * 0.9, kw: 0, exact: 0, total: s.total * 0.9 });
        }
      }
    }
    picked = [...top, ...extra];
    // 邻接补进来的按原顺序排在命中之后，不再重排（保持叙述连贯）
  }

  const evidence: RagEvidence[] = picked.map((s, i) => ({
    ref: `S${i + 1}`,
    chunkId: s.chunk.id,
    source: s.chunk.source,
    heading: s.chunk.heading,
    text: s.chunk.text,
    factLevel: s.chunk.factLevel as FactLevel,
    score: Number(s.cos.toFixed(4)),
    keywordBonus: Number((KEYWORD_WEIGHT * s.kw + EXACT_BONUS * Math.min(s.exact, 2)).toFixed(4)),
  }));

  const lowConfidence = topCos < minScore;

  // 证据契约（防瞎编的关键）：
  //   * topScore === 0 → **清空证据**。此时没有任何语义重合（维度不符已被跳过），
  //     留下的只是关键词噪声。若不清空，调用方写 `if (evidence.length)` 就会把
  //     完全无关的片段当依据注入提示词——这是一类静默的瞎编来源。
  //   * 0 < topScore < minScore → 保留证据（它是"桌上有什么料"的真实信息），
  //     但 lowConfidence=true，由调用方决定是否走兜底话术。
  //   * 自测模式（debug）保留全部候选，便于用户看到真实分数分布。
  const finalEvidence = opts.debug ? evidence : topCos > 0 ? evidence : [];

  return {
    evidence: finalEvidence,
    topScore: Number(topCos.toFixed(4)),
    lowConfidence,
    latencyMs: Date.now() - t0,
  };
}
