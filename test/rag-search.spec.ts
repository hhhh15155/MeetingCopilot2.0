/**
 * 检索层单元测试：余弦、中文字符 bigram 关键词、低置信负信号。
 * 负信号（lowConfidence）是防瞎编的地基，必须有测试守住。
 */
import { describe, expect, it } from 'vitest';
import { cosine, exactHits, keywordScore, search, tokenize, DEFAULT_MIN_SCORE } from '../electron/rag/search';
import type { RagChunk } from '../shared/protocol';

function mkChunk(partial: Partial<RagChunk> & { id: string; text: string; vector?: number[] }): RagChunk {
  return {
    source: '讲稿.md',
    slot: 'resume',
    index: 0,
    heading: '第1部分',
    factLevel: 'verified',
    ...partial,
  } as RagChunk;
}

describe('cosine', () => {
  it('相同向量为 1，正交为 0', () => {
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0);
  });
  it('维度不一致或零向量返回 0（不抛异常）', () => {
    expect(cosine([1, 0], [1, 0, 0])).toBe(0);
    expect(cosine([0, 0], [1, 0])).toBe(0);
  });
});

describe('tokenize / 中文 bigram', () => {
  it('中文取二字 bigram，无需分词器', () => {
    expect(tokenize('知识服务平台')).toEqual(['知识', '识服', '服务', '务平', '平台']);
  });
  it('拉丁词与数字指标原样保留', () => {
    const t = tokenize('FTS5 的 RRF 融合 91.3%');
    expect(t).toContain('fts5');
    expect(t).toContain('rrf');
    expect(t.some((x) => x.includes('91.3'))).toBe(true);
  });
});

describe('keywordScore / exactHits', () => {
  it('中文查询在无空格文档上能命中（NexQ 的 FTS5 在这里会 0 命中）', () => {
    const doc = '我负责海洋遥感多智能体知识服务平台的项目开发。';
    for (const q of ['海洋遥感', '知识服务平台', '多智能体']) {
      expect(keywordScore(tokenize(q), doc)).toBeGreaterThan(0.5);
    }
  });
  it('无关查询覆盖度为 0', () => {
    expect(keywordScore(tokenize('数据库索引优化'), '我负责前端界面的样式调整。')).toBe(0);
  });
  it('专有名词与指标能精确命中', () => {
    expect(exactHits('RRF 的 K 值取多少？91.3% 是怎么测的？', '用 RRF 融合，K=91.3%')).toBeGreaterThan(0);
  });
});

describe('search / 低置信负信号', () => {
  const store = {
    all: (): RagChunk[] => [
      mkChunk({ id: 'a', text: '我负责了 RAG 检索模块，用向量加 BM25 做混合召回。', vector: [1, 0, 0] }),
      mkChunk({ id: 'b', text: '前端界面用了 React 和 Tailwind 写样式。', vector: [0, 1, 0] }),
    ],
  };

  it('空索引 → 一定低置信（宁可说"没找到"，也不让模型自由发挥）', () => {
    const r = search({ all: () => [] }, '随便问', [1, 0, 0]);
    expect(r.lowConfidence).toBe(true);
    expect(r.evidence).toHaveLength(0);
    expect(r.topScore).toBe(0);
  });

  it('语义命中返回证据，分数为原始余弦', () => {
    const r = search(store, '你的检索是怎么做的', [1, 0, 0], { topK: 2 });
    expect(r.evidence.length).toBeGreaterThan(0);
    expect(r.evidence[0].chunkId).toBe('a');
    expect(r.evidence[0].score).toBeCloseTo(1, 3);
    expect(r.lowConfidence).toBe(false);
  });

  it('相似度不足 → lowConfidence=true（不让 LLM 接手）', () => {
    // 查询向量与两个片段都近乎正交
    const r = search(store, '完全不相关的问题', [0.01, 0.01, 1], { minScore: DEFAULT_MIN_SCORE });
    expect(r.topScore).toBeLessThan(DEFAULT_MIN_SCORE);
    expect(r.lowConfidence).toBe(true);
  });

  it('证据带编号 ref 与事实档位', () => {
    const r = search(store, '检索', [1, 0, 0], { topK: 2 });
    expect(r.evidence[0].ref).toBe('S1');
    expect(r.evidence[0].factLevel).toBe('verified');
  });

  it('邻接扩展把同源相邻片段补进来', () => {
    const s2 = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'a:0', index: 0, text: '第一段讲背景。', vector: [1, 0] }),
        mkChunk({ id: 'a:1', index: 1, text: '第二段讲实现细节。', vector: [0.9, 0.1] }),
      ],
    };
    const plain = search(s2, '背景', [1, 0], { topK: 1 });
    const expanded = search(s2, '背景', [1, 0], { topK: 1, neighborExpand: true });
    expect(plain.evidence.length).toBe(1);
    expect(expanded.evidence.length).toBe(2);
  });

  it('可按槽位过滤', () => {
    const mixed = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'r', text: '项目经历', vector: [1, 0], slot: 'resume' }),
        mkChunk({ id: 'j', text: '岗位要求', vector: [1, 0], slot: 'jd' }),
      ],
    };
    const r = search(mixed, '要求', [1, 0], { slots: ['jd'] });
    expect(r.evidence.every((e) => e.chunkId === 'j')).toBe(true);
  });
});
