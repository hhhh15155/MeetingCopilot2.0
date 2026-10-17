/**
 * 引用覆盖率审计的单元测试。
 *
 * 这是"防瞎编"的最后一道可验证环节：模型说没说、说的有没有出处，
 * 必须能当场看出来。规则是确定性的，所以测试必须锁死它的边界行为
 * （尤其是"不要因为泛化陈述而误报"——满屏告警等于没有告警）。
 */
import { describe, expect, it } from 'vitest';
import {
  auditCitations,
  needsCitation,
  parseCitations,
  splitAnswerSentences,
} from '../shared/citationAudit';
import type { RagEvidence } from '../shared/protocol';

function ev(ref: string, text = '内容'): RagEvidence {
  return {
    ref,
    chunkId: `c:${ref}`,
    source: '讲稿.md',
    heading: '第1部分',
    text,
    factLevel: 'verified',
    score: 0.8,
    keywordBonus: 0,
  };
}

describe('splitAnswerSentences', () => {
  it('按行切分并剥离列表符号与加粗', () => {
    const s = splitAnswerSentences('- **我在项目里做检索。**\n- 用了 RRF 融合。');
    expect(s).toEqual(['我在项目里做检索。', '用了 RRF 融合。']);
  });
  it('行内按句末标点细分，但不以逗号切', () => {
    const s = splitAnswerSentences('我做的 XX 项目，召回率 88.7%。第二句。');
    expect(s).toHaveLength(2);
    // 关键：逗号两侧没有被拆开（否则会把"项目"和"88.7%"分到两句造成误报）
    expect(s[0]).toContain('项目');
    expect(s[0]).toContain('88.7%');
  });
  it('空输入返回空数组', () => {
    expect(splitAnswerSentences('')).toEqual([]);
    expect(splitAnswerSentences('\n\n  \n')).toEqual([]);
  });
});

describe('parseCitations', () => {
  it('识别多种编号写法', () => {
    expect(parseCitations('基于 [S1] 的说法')).toEqual(['S1']);
    expect(parseCitations('【S2】里有')).toEqual(['S2']);
    expect(parseCitations('见 [S1][S3]')).toEqual(['S1', 'S3']);
    expect(parseCitations('合并 [S1, S2] 与 [s3 s4]')).toEqual(['S1', 'S2', 'S3', 'S4']);
  });
  it('不把普通方括号内容当引用', () => {
    expect(parseCitations('这是 [备注] 不是引用')).toEqual([]);
    expect(parseCitations('[2024] 年')).toEqual([]);
  });
  it('同一句重复引用只算一次', () => {
    expect(parseCitations('[S1] 和 [S1]')).toEqual(['S1', 'S1']); // 去重由 audit 负责
  });
});

describe('needsCitation', () => {
  it('含数字的句子需要出处', () => {
    expect(needsCitation('召回率是 88.7%。')).toBe(true);
    expect(needsCitation('有 3 个 Agent。')).toBe(true);
  });
  it('含公司/平台/项目等具体指称需要出处', () => {
    expect(needsCitation('我在海洋遥感平台做检索。')).toBe(true);
    expect(needsCitation('用的是 ChromaDB。')).toBe(true);
  });
  it('泛化陈述不需要出处（避免满屏告警）', () => {
    expect(needsCitation('我觉得沟通在这个岗位上很重要。')).toBe(false);
    expect(needsCitation('我会主动把问题拆解清楚。')).toBe(false);
  });
});

describe('auditCitations', () => {
  const evidence = [ev('S1'), ev('S2')];

  it('全部事实句都有引用 → 覆盖率 100%', () => {
    const a = auditCitations('召回率 88.7% [S1]。\n用了 RRF 融合 [S2]。', evidence);
    expect(a.cited).toEqual(['S1', 'S2']);
    expect(a.uncitedFacts).toEqual([]);
    expect(a.coverage).toBe(1);
    expect(a.factCount).toBe(2);
  });

  it('事实句缺引用 → 被列出且覆盖率下降', () => {
    const a = auditCitations('召回率 88.7% [S1]。\n测试集有 500 条。', evidence);
    expect(a.cited).toEqual(['S1']);
    expect(a.uncitedFacts).toEqual(['测试集有 500 条。']);
    expect(a.factCount).toBe(2);
    expect(a.coverage).toBe(0.5);
  });

  it('引用不存在的编号 → 记入 dangling（编造来源）', () => {
    const a = auditCitations('我在阿里云做过这个 [S9]。', evidence);
    expect(a.dangling).toEqual(['S9']);
    expect(a.cited).toEqual([]); // S9 没被提供过，不算有效引用
    // 该句仍算"没有有效引用"的事实句
    expect(a.uncitedFacts).toHaveLength(1);
  });

  it('纯泛化陈述：没有事实句时不误报为"没引用"', () => {
    const a = auditCitations('我会先讲结论，再展开要点。', evidence);
    expect(a.factCount).toBe(0);
    expect(a.coverage).toBe(1);
    expect(a.uncitedFacts).toEqual([]);
  });

  it('无证据时任何事实句都算缺引用', () => {
    const a = auditCitations('召回率 88.7%。', []);
    expect(a.coverage).toBe(0);
    expect(a.uncitedFacts).toHaveLength(1);
  });

  it('cited 按编号数字升序、去重', () => {
    const a = auditCitations('[S3] 与 [S1] 与 [S3]', [ev('S1'), ev('S3')]);
    expect(a.cited).toEqual(['S1', 'S3']);
  });

  it('空回答不崩', () => {
    const a = auditCitations('', evidence);
    expect(a).toEqual({ cited: [], dangling: [], uncitedFacts: [], factCount: 0, coverage: 1 });
  });

  it('中文列表式回答逐行核对（真实形状）', () => {
    const answer = [
      '我做过一个海洋遥感知识服务平台，用的是多智能体架构 [S1]。',
      '检索部分是向量加 BM25 混合召回，用 RRF 融合，K 取 60 [S2]。',
      '另外我还顺手做了模型蒸馏，把参数量压到了 1.5B。',
      '整体上我比较注重工程落地。',
    ].join('\n');
    const a = auditCitations(answer, evidence);
    expect(a.cited).toEqual(['S1', 'S2']);
    // 第三句含数字且无引用 → 必须被指出
    expect(a.uncitedFacts).toHaveLength(1);
    expect(a.uncitedFacts[0]).toContain('1.5B');
    // 第四句是泛化陈述 → 不进统计
    expect(a.factCount).toBe(3);
    expect(a.coverage).toBeCloseTo(2 / 3, 5);
  });
});
