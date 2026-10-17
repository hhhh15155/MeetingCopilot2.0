/**
 * RAG 门面：把切块/embedding/存储/检索/索引打包成主进程可直接用的服务。
 *
 * 主进程只需要：
 *   const rag = createRagService(索引文件路径, () => 当前 embedding 配置, () => apiKey)
 *   await rag.indexDocument({slot, source, text})
 *   const r = await rag.retrieve(question)
 * 其余细节全在这里收口，便于单测与替换实现。
 */
import type { KbSlot, RagIndexProgress, RagRetrieval } from '../../shared/protocol';
import { embedQuery, embedTexts, testEmbedding, type EmbeddingSettings } from './embedder';
import { indexDocument, type IndexResult } from './indexer';
import { search, DEFAULT_MIN_SCORE } from './search';
import { RagStore } from './store';

export { RagStore } from './store';
export { chunkMarkdown, chunkLabel } from './chunker';
export { classifyChunk, classifySentence, factLevelBreakdown, FACT_LEVEL_LABEL, FACT_LEVEL_RULE } from './factlevel';
export { embedTexts, embedQuery, testEmbedding, EMBEDDING_PRESETS, DEFAULT_EMBEDDING_PRESET, embeddingsEndpoint } from './embedder';
export type { EmbeddingSettings, EmbeddingResult, EmbeddingProviderPreset } from './embedder';
export { indexDocument } from './indexer';
export type { IndexResult } from './indexer';
export { search, cosine, tokenize, DEFAULT_MIN_SCORE } from './search';

export interface RagRuntimeConfig {
  enabled: boolean;
  topK: number;
  minScore: number;
  neighborExpand: boolean;
  requireCitations: boolean;
  embedding: EmbeddingSettings;
}

export const DEFAULT_RAG_CONFIG: RagRuntimeConfig = {
  enabled: true,
  topK: 5,
  minScore: DEFAULT_MIN_SCORE,
  neighborExpand: true,
  requireCitations: true,
  embedding: {
    providerId: 'dashscope',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'text-embedding-v3',
    dim: 1024,
    batchSize: 16,
  },
};

export class RagService {
  constructor(
    private readonly store_: RagStore,
    private readonly getConfig: () => RagRuntimeConfig,
    private readonly getApiKey: () => string,
  ) {}

  /**
   * 底层存储（供主进程做 info/clear 等只读操作）。
   * 注意不能叫 `index` —— 与下面的 `index()` 方法同名会导致 TS2300
   * Duplicate identifier（getter 与方法共享同一命名空间）。
   */
  get store(): RagStore {
    return this.store_;
  }

  info() {
    return this.store.info();
  }

  /** 索引进度回调（由主进程桥到渲染进程） */
  onProgress: ((p: RagIndexProgress) => void) | undefined;

  async index(opts: {
    slot: KbSlot;
    source: string;
    text: string;
    signal?: AbortSignal;
  }): Promise<IndexResult> {
    const cfg = this.getConfig();
    const apiKey = this.getApiKey();
    if (!apiKey) throw new Error('未配置 embedding API Key，无法建立索引');
    // 换了 embedding 模型 → 旧向量不可比，必须先清空重建
    if (this.store.size > 0 && !this.store.matchesModel(cfg.embedding.model)) {
      this.store.clear();
    }
    return indexDocument(this.store, embedTexts, {
      slot: opts.slot,
      source: opts.source,
      text: opts.text,
      embedding: cfg.embedding,
      apiKey,
      signal: opts.signal,
      onProgress: (p) => this.onProgress?.(p),
    });
  }

  /**
   * 运行时检索（热路径）：只调一次 query embedding，然后进程内余弦。
   * 任何一步失败都返回"低置信空结果"，绝不静默退回无约束的全量直塞。
   */
  async retrieve(question: string, opts: { slot?: KbSlot; signal?: AbortSignal } = {}): Promise<RagRetrieval> {
    const t0 = Date.now();
    const cfg = this.getConfig();
    const empty: RagRetrieval = { evidence: [], topScore: 0, lowConfidence: true, latencyMs: 0 };
    if (!cfg.enabled) return empty;
    if (this.store.size === 0) return { ...empty, latencyMs: Date.now() - t0 };
    if (this.store.modelId && !this.store.matchesModel(cfg.embedding.model)) {
      // 索引与当前模型不匹配：当作没索引，让用户重建，而不是给出错误相似度
      return { ...empty, latencyMs: Date.now() - t0 };
    }
    const apiKey = this.getApiKey();
    if (!apiKey) return { ...empty, latencyMs: Date.now() - t0 };
    const q = question.trim();
    if (!q) return { ...empty, latencyMs: Date.now() - t0 };

    let vec: number[];
    try {
      vec = await embedQuery(cfg.embedding, apiKey, q, opts.signal);
    } catch {
      return { ...empty, latencyMs: Date.now() - t0 };
    }
    const r = search(this.store, q, vec, {
      topK: cfg.topK,
      minScore: cfg.minScore,
      neighborExpand: cfg.neighborExpand,
      slots: opts.slot ? [opts.slot] : undefined,
    });
    return { ...r, latencyMs: Date.now() - t0 };
  }

  /** 检索自测：走同一套逻辑，供设置面板"试一下能不能命中"。 */
  async searchTest(question: string, topK?: number): Promise<RagRetrieval> {
    const cfg = this.getConfig();
    const apiKey = this.getApiKey();
    if (!apiKey) throw new Error('未配置 embedding API Key');
    const vec = await embedQuery(cfg.embedding, apiKey, question);
    return search(this.store, question, vec, {
      topK: topK ?? cfg.topK,
      minScore: 0, // 自测时不滤掉低分，便于用户看到真实分数分布
      neighborExpand: false,
      debug: true, // 自测要看到全部候选，不能被"低置信清空证据"滤掉
    });
  }

  clear(): void {
    this.store.clear();
  }

  async testConnection(): Promise<{ ok: boolean; messageZh: string; dim?: number; latencyMs?: number }> {
    const cfg = this.getConfig();
    return testEmbedding(cfg.embedding, this.getApiKey());
  }
}

export function createRagService(
  indexPath: string,
  getConfig: () => RagRuntimeConfig,
  getApiKey: () => string,
): RagService {
  return new RagService(new RagStore(indexPath), getConfig, getApiKey);
}
