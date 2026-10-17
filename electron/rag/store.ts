/**
 * 索引存储：JSON 文件 + 进程内暴力余弦。
 *
 * 为什么不用 SQLite（NexQ 的做法）：NexQ 用 SQLite FTS5 + BLOB 向量 +
 * 3 个同步触发器。搬到 Electron 需要 better-sqlite3（原生模块，要 rebuild、
 * 影响打包链）或 Node 22 的 node:sqlite。而我们的语料只有 11 万字量级
 * （≈300 片段 × 1024 维 × 4B ≈ 1.2MB），**全量载入内存暴力算余弦是毫秒级**，
 * 完全不需要数据库，更不需要 ANN 索引。
 *
 * 顺带避开 NexQ 的一个坑：它的 FTS5 建表没指定分词器，中文整段被当成
 * 一个巨大 token，实测"海洋遥感""知识服务平台"等查询全部 0 命中——
 * 混合检索的 keyword 分支实际是死的。本实现改用**字符 bigram 召回**
 * （中文无需分词器），并且分数是原始余弦，阈值可解释。
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'fs';
import { dirname } from 'path';
import type { FactLevel, KbSlot, RagChunk } from '../../shared/protocol';

export interface RagIndexFile {
  /** 格式版本，便于日后迁移 */
  version: 1;
  /** embedding 模型标识；更换模型后此索引作废，必须重建 */
  model: string;
  dim: number;
  indexedAt: number;
  chunks: RagChunk[];
}

export class RagStore {
  private chunks: RagChunk[] = [];
  private model = '';
  private dim = 0;
  private indexedAt = 0;

  constructor(private readonly filePath: string) {
    this.load();
  }

  private load(): void {
    try {
      if (!existsSync(this.filePath)) return;
      const raw = JSON.parse(readFileSync(this.filePath, 'utf8')) as RagIndexFile;
      if (raw?.version !== 1 || !Array.isArray(raw.chunks)) return;
      this.chunks = raw.chunks;
      this.model = raw.model ?? '';
      this.dim = raw.dim ?? 0;
      this.indexedAt = raw.indexedAt ?? 0;
    } catch {
      // 索引损坏不应让应用起不来：当作空索引，用户重建即可
      this.chunks = [];
      this.model = '';
      this.dim = 0;
      this.indexedAt = 0;
    }
  }

  /** 原子落盘：先写临时文件再 rename，避免中断留下半个文件。 */
  persist(): void {
    const dir = dirname(this.filePath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    const payload: RagIndexFile = {
      version: 1,
      model: this.model,
      dim: this.dim,
      indexedAt: this.indexedAt,
      chunks: this.chunks,
    };
    const tmp = `${this.filePath}.tmp`;
    writeFileSync(tmp, JSON.stringify(payload), 'utf8');
    renameSync(tmp, this.filePath);
  }

  get size(): number {
    return this.chunks.length;
  }
  get modelId(): string {
    return this.model;
  }
  get dimension(): number {
    return this.dim;
  }
  get indexedTime(): number {
    return this.indexedAt;
  }
  /** 全部片段（检索用；返回只读引用，调用方不得修改） */
  all(): readonly RagChunk[] {
    return this.chunks;
  }

  /** 索引总览，供界面显示。 */
  info(): { chunkCount: number; sources: { name: string; slot: KbSlot; chunks: number }[]; indexedAt?: number; model?: string; dim?: number } {
    const bySource = new Map<string, { name: string; slot: KbSlot; chunks: number }>();
    for (const c of this.chunks) {
      const key = `${c.slot}::${c.source}`;
      const cur = bySource.get(key);
      if (cur) cur.chunks++;
      else bySource.set(key, { name: c.source, slot: c.slot, chunks: 1 });
    }
    return {
      chunkCount: this.chunks.length,
      sources: [...bySource.values()],
      indexedAt: this.indexedAt || undefined,
      model: this.model || undefined,
      dim: this.dim || undefined,
    };
  }

  /** 取某来源的片段（邻接扩展用）。 */
  bySource(source: string, slot: KbSlot): RagChunk[] {
    return this.chunks.filter((c) => c.source === source && c.slot === slot);
  }
  byId(id: string): RagChunk | undefined {
    return this.chunks.find((c) => c.id === id);
  }

  /** 单个槽位的所有片段（重建前先取出，便于保留另一槽）。 */
  forSlot(slot: KbSlot): RagChunk[] {
    return this.chunks.filter((c) => c.slot === slot);
  }

  /**
   * 用新片段替换某个槽位的内容（另一槽保持不变）。
   * 同一槽位 + 同一来源名视为"这份资料被重建"，旧片段整体替换。
   */
  replaceSlot(
    slot: KbSlot,
    source: string,
    slots: { text: string; heading: string; index: number }[],
    factLevels: FactLevel[],
    vectors: number[][],
    model: string,
  ): void {
    this.chunks = this.chunks.filter((c) => !(c.slot === slot && c.source === source));
    slots.forEach((s, i) => {
      this.chunks.push({
        id: `${slot}:${source}:${s.index}`,
        source,
        slot,
        index: s.index,
        heading: s.heading,
        text: s.text,
        factLevel: factLevels[i] ?? 'verified',
        vector: vectors[i],
      });
    });
    this.model = model;
    this.dim = vectors[0]?.length ?? this.dim;
    this.indexedAt = Date.now();
  }

  clear(): void {
    this.chunks = [];
    this.model = '';
    this.dim = 0;
    this.indexedAt = 0;
    try {
      if (existsSync(this.filePath)) writeFileSync(this.filePath, JSON.stringify({ version: 1, model: '', dim: 0, indexedAt: 0, chunks: [] } satisfies RagIndexFile), 'utf8');
    } catch {
      /* ignore */
    }
  }

  /** 索引是否与当前 embedding 模型匹配；不匹配必须重建（维度/语义空间都变了）。 */
  matchesModel(model: string): boolean {
    return this.model === model;
  }
}
