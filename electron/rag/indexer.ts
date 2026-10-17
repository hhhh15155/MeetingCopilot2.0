/**
 * 索引器：把一份资料文本切成片段 → 标注事实档位 → 算 embedding → 落库。
 *
 * 索引是**一次性离线任务**（面试前点一次"重建索引"），运行时热路径只有
 * 一次 query embedding。因此这里可以慢、可以重试，但必须：可复现、可中断、
 * 有进度、失败不静默。
 *
 * embedder 以函数形式注入，便于单测不联网。
 */
import type { FactLevel, KbSlot, RagIndexProgress } from '../../shared/protocol';
import { chunkMarkdown, type ChunkDraft, type ChunkOptions } from './chunker';
import { classifyChunk } from './factlevel';
import type { EmbeddingSettings } from './embedder';
import { RagStore } from './store';

export type EmbedFn = (
  settings: EmbeddingSettings,
  apiKey: string,
  texts: string[],
  opts?: { signal?: AbortSignal; onProgress?: (done: number, total: number) => void },
) => Promise<{ vectors: number[][]; dim: number; latencyMs: number }>;

export interface IndexOptions {
  slot: KbSlot;
  source: string;
  text: string;
  embedding: EmbeddingSettings;
  apiKey: string;
  chunk?: ChunkOptions;
  signal?: AbortSignal;
  onProgress?: (p: RagIndexProgress) => void;
}

export interface IndexResult {
  chunks: number;
  dim: number;
  latencyMs: number;
  /** 各档位片段数，注册表可用它提示用户"你这份资料里有多少是包装值" */
  byLevel: Record<FactLevel, number>;
}

/**
 * 重建某个槽位下某份资料的索引。
 * 注意：如果索引里已有的模型与当前 embedding 模型不同，会在调用方被拒绝
 * （不同模型的向量空间不可比），这里只负责按当前模型重建。
 */
export async function indexDocument(
  store: RagStore,
  embed: EmbedFn,
  opts: IndexOptions,
): Promise<IndexResult> {
  const { slot, source, text, embedding, apiKey } = opts;
  const progress = (p: RagIndexProgress) => opts.onProgress?.(p);

  progress({ slot, status: 'chunking', source, chunksTotal: 0, chunksEmbedded: 0 });

  const drafts: ChunkDraft[] = chunkMarkdown(text, opts.chunk);
  if (drafts.length === 0) {
    progress({ slot, status: 'error', source, chunksTotal: 0, chunksEmbedded: 0, error: '资料内容为空，未提取到任何文本' });
    return { chunks: 0, dim: 0, latencyMs: 0, byLevel: { verified: 0, claimed: 0, unverified: 0 } };
  }

  // 事实档位先算好：这是防瞎编的关键元数据，必须随片段一起入库。
  //
  // ⚠️ 只判**正文**，不能把标题路径一起判：标题是导航信息而非事实陈述，
  // 而且顶层标题会被带进每一个片段。实测踩过：某文档标题含"证明"二字，
  // 恰好命中边界词表，导致那份 94 段的手册**全篇 100% 被标成"待确认"**，
  // 徽章彻底失去意义（用户脱敏 → 真正的包装数字反而被淹没）。
  const byLevel: Record<FactLevel, number> = { verified: 0, claimed: 0, unverified: 0 };
  const levels: FactLevel[] = drafts.map((d) => {
    const lv = classifyChunk(d.text);
    byLevel[lv]++;
    return lv;
  });

  // 入库的文本带上标题路径：标题词是检索的强信号，也让模型知道这段话的出处
  const payloads = drafts.map((d) => (d.heading ? `【${d.heading}】\n${d.text}` : d.text));

  progress({ slot, status: 'embedding', source, chunksTotal: drafts.length, chunksEmbedded: 0 });

  try {
    const r = await embed(embedding, apiKey, payloads, {
      signal: opts.signal,
      onProgress: (done, total) =>
        progress({ slot, status: 'embedding', source, chunksTotal: total, chunksEmbedded: done }),
    });
    if (r.vectors.length !== drafts.length) {
      throw new Error(`向量条数不符：期望 ${drafts.length}，实际 ${r.vectors.length}`);
    }
    // 纵深防御 1：不允许带洞的向量数组入库。
    // 带洞片段会被静默写进索引、永远检索不到，且不报任何错——最隐蔽的一类故障。
    const hole = r.vectors.findIndex((v) => !v || v.length === 0);
    if (hole >= 0) throw new Error(`第 ${hole} 条向量缺失，索引未完整生成（已中止入库）`);

    // 纵深防御 2：**同一索引内维度必须一致**。
    // 用户把 dim 从 1024 改成 512 后重建一份资料，若混入不同维度，
    // 另一份资料的片段余弦恒为 0、永远检索不到且零提示。宁可整批失败。
    const dims = new Set(r.vectors.map((v) => v.length));
    if (dims.size > 1) {
      throw new Error(
        `向量维度不一致（${[...dims].join(' / ')}）：embedding 服务返回了混杂维度，` +
          `已中止入库以免产生永远检索不到的片段`,
      );
    }
    const newDim = r.vectors[0].length;
    if (store.size > 0 && store.dimension > 0 && store.dimension !== newDim) {
      throw new Error(
        `新向量维度 ${newDim} 与现有索引维度 ${store.dimension} 不一致。` +
          `请先清空索引再重建（混用维度会让部分片段永远检索不到）`,
      );
    }

    const chunks = drafts.map((d, i) => ({ text: payloads[i], heading: d.heading, index: d.index }));
    store.replaceSlot(slot, source, chunks, levels, r.vectors, embedding.model);
    store.persist();
    progress({ slot, status: 'ready', source, chunksTotal: drafts.length, chunksEmbedded: drafts.length });
    return { chunks: drafts.length, dim: r.dim, latencyMs: r.latencyMs, byLevel };
  } catch (e) {
    const msg = (e as Error).message;
    progress({ slot, status: 'error', source, chunksTotal: drafts.length, chunksEmbedded: 0, error: msg });
    throw e;
  }
}
