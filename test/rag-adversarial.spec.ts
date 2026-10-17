/**
 * 对抗性验证 spec —— 独立验证者（rag-verifier）编写，**不修改**被测实现。
 *
 * 约定（写给 Lead 与后来者）：
 *   * 本文件里的断言写的是「期望的正确行为」。**用例失败 = 实现缺陷**，不是测试写错了。
 *   * 判定类用例（factlevel）用红/绿直接暴露漏判；
 *     纯记录类结论用 console.log 输出，供报告引用，不做对错断言。
 *   * 需要联网的路径全部 mock（globalThis.fetch）；只有 test/../ 之外不碰任何文件。
 *
 * 覆盖面：事实边界漏判 / 切块丢内容 / 切块破坏问答对 / 检索负信号 /
 *        维度不匹配 / embedder 端点规范化与错误处理 / store 落盘 / indexer 集成。
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { afterAll, afterEach, describe, expect, it, vi } from 'vitest';
import { chunkMarkdown, splitSections, type ChunkDraft } from '../electron/rag/chunker';
import { classifyChunk, classifySentence, factLevelBreakdown } from '../electron/rag/factlevel';
import { cosine, DEFAULT_MIN_SCORE, exactHits, keywordScore, search, tokenize } from '../electron/rag/search';
import { embedTexts, embeddingsEndpoint, type EmbeddingSettings } from '../electron/rag/embedder';
import { indexDocument } from '../electron/rag/indexer';
import { RagStore } from '../electron/rag/store';
import type { FactLevel, RagChunk } from '../shared/protocol';

// ============================================================================
// 0. 工具
// ============================================================================

/** 重建 chunk 文本（含重叠前缀）——用于内容保全检查。 */
const joinChunks = (out: readonly ChunkDraft[]): string => out.map((c) => c.text).join('\n');
/** 去掉空白与 markdown 结构字符后比较，避免因标记差异造成假丢内容。 */
const norm = (s: string): string => s.replace(/[\s#`|>…]+/g, '');
/** 去掉粗体/引用/空白，便于识别「问：/答：」标记（**问**：也要算）。 */
const stripMark = (s: string): string => s.replace(/[*>`\s]+/g, '');

/** n-gram 覆盖率：源文每 n 个字组成的片段有多少比例出现在切块结果里。 */
function coverage(src: string, out: readonly ChunkDraft[], n = 8): { ratio: number; miss: string[] } {
  const a = norm(src);
  const b = norm(joinChunks(out));
  let total = 0;
  const miss: string[] = [];
  for (let i = 0; i + n <= a.length; i++) {
    total++;
    const g = a.slice(i, i + n);
    if (!b.includes(g)) miss.push(g);
  }
  return { ratio: total === 0 ? 1 : 1 - miss.length / total, miss: miss.slice(0, 8) };
}

/** 问答对完整性：片段里出现答句标记 A{i}_ 却没有对应问题标记 QAQ{i}，即「只有答没有问」。 */
function qaViolations(out: readonly ChunkDraft[], pairs: number): string[] {
  const bad: string[] = [];
  for (const c of out) {
    for (let i = 1; i <= pairs; i++) {
      if (c.text.includes(`A${i}_`) && !c.text.includes(`QAQ${i}`)) {
        bad.push(`第${c.index + 1}段含答${i}但无问${i}：「${stripMark(c.text).slice(0, 28)}…」`);
      }
    }
  }
  return bad;
}

function mkChunk(p: Partial<RagChunk> & { id: string; text: string }): RagChunk {
  return { source: 'a.md', slot: 'resume', index: 0, heading: 'h', factLevel: 'verified', ...p } as RagChunk;
}

// ============================================================================
// 1. 事实边界：漏判探针（红 = 包装数字被判成 verified，防瞎编失效）
// ============================================================================

interface LevelCase {
  name: string;
  sent: string;
  want: FactLevel;
  why: string;
}

/** 必须被判 claimed（包装值）——被判成 verified 就是漏判。 */
const MUST_BE_CLAIMED: LevelCase[] = [
  {
    name: '基线：句内百分比 + 无简历字样',
    sent: '我在 XX 项目里把准确率从 78% 提到了 91.3%。',
    want: 'claimed',
    why: 'METRIC_RE 命中「78%」，无仓库词 → claimed（当前实现正确）',
  },
  {
    name: '条数',
    sent: '测试集规模 500 条，全部通过。',
    want: 'claimed',
    why: 'METRIC_RE 命中「500 条」→ claimed（当前实现正确）',
  },
  {
    name: '百分比 + 百分点',
    sent: 'RAG 召回率 88.7%，比基线高 12 个百分点。',
    want: 'claimed',
    why: '当前实现正确',
  },
  {
    name: '显式【项目】+ 指标',
    sent: '【项目】准确率 91.3%。',
    want: 'claimed',
    why: '有指标就必须保守；注意 EXPLICIT_VERIFIED 在有指标时被第 140 行短路，等于失效',
  },
  {
    name: '漏判探针①：无单位小数（F1）',
    sent: 'Macro-F1 达到 0.89。',
    want: 'claimed',
    why: 'METRIC_RE 要求数字后必须跟 %/条/个/倍/ms…，「0.89」不带单位 → 落到 verified',
  },
  {
    name: '漏判探针①：无单位整数量（QPS）',
    sent: '我把 QPS 从 300 提到了 1200。',
    want: 'claimed',
    why: '同上：裸数字不算 metric',
  },
  {
    name: '漏判探针①：无单位 baseline',
    sent: 'baseline 准确率 81.8。',
    want: 'claimed',
    why: '同上',
  },
  {
    name: '漏判探针②：句内出现「模块」',
    sent: '我负责的模块准确率 91.3%。',
    want: 'claimed',
    why: 'REPO_EVIDENCE 含 /模块/ → hasRepo=true → 第 140 行跳过 claimed → verified',
  },
  {
    name: '漏判探针②：句子含「类」（类似/人类/分类）',
    sent: '我用类似的方法把准确率做到了 91.3%。',
    want: 'claimed',
    why: 'REPO_EVIDENCE 的 /类/ 是单字匹配，任何含「类」的句子都被当成有仓库证据',
  },
  {
    name: '漏判探针②：句内出现文件名',
    sent: '在 src/rag/search.ts 里把召回率调到了 92%。',
    want: 'claimed',
    why: '提到文件不等于这个数字有实测支撑 — 但实现把它当 verified',
  },
  {
    name: '漏判探针②：句内出现「实现了」',
    sent: '我实现了 500 条测试集的自动评测，准确率 91%。',
    want: 'claimed',
    why: '/（实现|完成）了/ 命中 → verified',
  },
  {
    name: '漏判探针②：句内出现「接口」',
    sent: '数据接口的 QPS 从 200 提到了 1500。',
    want: 'claimed',
    why: '「接口」是普通技术名词，被当成仓库证据',
  },
  {
    name: '漏判探针②：「人类」中的 类',
    sent: '人类标注样本上的准确率是 92.4%。',
    want: 'claimed',
    why: '同上',
  },
];

/** 反向对照：不能被过度保守地降级（否则模型该讲的经历也不敢讲）。 */
const MUST_NOT_BE_CLAIMED: LevelCase[] = [
  { name: '规划口吻', sent: '后续计划把召回率提到 95%。', want: 'unverified', why: '规划不得讲成已完成' },
  { name: '预期口吻', sent: '预计吞吐提升 3 倍。', want: 'unverified', why: '同上' },
  { name: '可指向代码', sent: '在 electron/rag/search.ts 里实现了 RRF 融合。', want: 'verified', why: '无指标 + 有文件 → verified 正确' },
  { name: '普通经历无数字', sent: '我负责了数据清洗和特征工程这两块工作。', want: 'verified', why: '无指标无规划 → verified 正确' },
  { name: '【边界】人工标记', sent: '【边界】这部分我没有实际做过。', want: 'claimed', why: '显式标记优先' },
];

describe('A. 事实边界漏判探针（红 = 漏判，包装数字会被当既成事实讲出去）', () => {
  it.each(MUST_BE_CLAIMED)('$name → 期望 claimed', (c) => {
    expect(classifySentence(c.sent), `${c.sent}  ← ${c.why}`).toBe(c.want);
  });

  it.each(MUST_NOT_BE_CLAIMED)('$name → 期望 $want（不得过度保守）', (c) => {
    expect(classifySentence(c.sent), `${c.sent}  ← ${c.why}`).toBe(c.want);
  });

  it('整块取最高风险：指标句与仓库句分属不同句子时必须仍为 claimed', () => {
    const mixed = ['我在 src/a.ts 里实现了检索模块。', '准确率 92.4%，测试集 500 条。'].join('\n');
    expect(classifyChunk(mixed)).toBe('claimed');
  });

  it('整块取最高风险：包装说明在**别的片段**时，含数字片段会丢掉 claimed（片段级标注的固有缺口）', () => {
    // 记录性断言：这是设计缺口，不是崩溃。索引时按片段独立判定，
    // 「简历里的…」这句被切到上一块后，下一块只剩数字 → verified。
    const prev = classifyChunk('简历里的 500 条 / Accuracy 91.3% 在仓库中没有支撑。');
    const next = classifyChunk('Macro-F1 0.89 是我最好的结果');
    expect(prev).toBe('claimed');
    // 修复后（Lead 2026-10-17）：含数字/指标的句子一律 claimed，
    // 即使它自己没写"简历里"。原记录为 verified（无单位数字漏判），已修正。
    expect(next).toBe('claimed');
  });
});

describe('B. 事实边界判定矩阵（诊断输出，供报告引用，不做对错断言）', () => {
  it('打印 classifySentence 实际结果', () => {
    const rows: [string, string][] = [
      ['我在 XX 项目里把准确率从 78% 提到了 91.3%', classifySentence('我在 XX 项目里把准确率从 78% 提到了 91.3%。')],
      ['测试集规模 500 条，全部通过', classifySentence('测试集规模 500 条，全部通过。')],
      ['RAG 召回率 88.7%，比基线高 12 个百分点', classifySentence('RAG 召回率 88.7%，比基线高 12 个百分点。')],
      ['【项目】准确率 91.3%', classifySentence('【项目】准确率 91.3%。')],
      ['【项目】我负责的检索模块，准确率 91.3%', classifySentence('【项目】我负责的检索模块，准确率 91.3%。')],
      ['Macro-F1 达到 0.89', classifySentence('Macro-F1 达到 0.89。')],
      ['我把 QPS 从 300 提到了 1200', classifySentence('我把 QPS 从 300 提到了 1200。')],
      ['我负责的模块准确率 91.3%', classifySentence('我负责的模块准确率 91.3%。')],
      ['我用类似的方法把准确率做到了 91.3%', classifySentence('我用类似的方法把准确率做到了 91.3%。')],
      ['人类标注样本上的准确率是 92.4%', classifySentence('人类标注样本上的准确率是 92.4%。')],
      ['在 src/rag/search.ts 里把召回率调到了 92%', classifySentence('在 src/rag/search.ts 里把召回率调到了 92%。')],
      ['我实现了 500 条测试集的自动评测，准确率 91%', classifySentence('我实现了 500 条测试集的自动评测，准确率 91%。')],
      ['简历里的 500 条 / Accuracy 91.3% 没有支撑', classifySentence('简历里的 500 条 / Accuracy 91.3% 在仓库中没有支撑。')],
      ['后续计划把召回率提到 95%', classifySentence('后续计划把召回率提到 95%。')],
      ['预计吞吐提升 3 倍', classifySentence('预计吞吐提升 3 倍。')],
      ['在 electron/rag/search.ts 里实现了 RRF 融合', classifySentence('在 electron/rag/search.ts 里实现了 RRF 融合。')],
    ];
    console.log('\n| # | 句子 | 实际判定 |\n| --- | --- | --- |\n' + rows.map(([s, lv], i) => `| ${i + 1} | ${s} | **${lv}** |`).join('\n'));
    const b = factLevelBreakdown('预计吞吐能提升 3 倍。\n我把 QPS 从 300 提到了 1200。');
    console.log(`breakdown 示例：${JSON.stringify(b)}（含裸数字句是否为 claimed 见上表）`);
    expect(rows).toHaveLength(16);
  });
});

// ============================================================================
// 2. 切块：内容保全
// ============================================================================

const DOC_ADVERSARIAL = [
  '# 项目文档',
  '',
  '## 1. 代码片段',
  '',
  '```python',
  '# 计算准确率',
  'def acc(y, p):',
  '    return sum(1 for a, b in zip(y, p) if a == b) / len(y)',
  '```',
  '',
  '## 2. 实验数据表',
  '',
  '| 数据集 | 样本数 | 准确率 |',
  '| --- | --- | --- |',
  '| 测试集 | 500 | 91.3% |',
  '',
  '## 3. 无标点超长段',
  '',
  '甲'.repeat(1600),
  '',
  '## 4. 只有标题没有正文的小节',
  '## 5. 另一个只有标题的小节',
].join('\n');

describe('C. 切块内容保全（不允许整段丢失）', () => {
  it('代码块 + 表格 + 超长无标点段 + 只有标题的小节：默认参数下 n-gram 覆盖率应 ≥ 0.98', () => {
    const out = chunkMarkdown(DOC_ADVERSARIAL);
    const cov = coverage(DOC_ADVERSARIAL, out);
    console.log(
      `默认参数（700/120）：${out.length} 段，覆盖率 ${(cov.ratio * 100).toFixed(2)}%，缺失片段 ${JSON.stringify(cov.miss)}`,
    );
    expect(cov.ratio, `缺失 n-gram: ${JSON.stringify(cov.miss)}`).toBeGreaterThanOrEqual(0.98);
  });

  it('小 chunkChars + 无重叠时同样不丢内容', () => {
    const out = chunkMarkdown(DOC_ADVERSARIAL, { chunkChars: 120, overlapChars: 0 });
    const cov = coverage(DOC_ADVERSARIAL, out);
    console.log(
      `小参数（120/0）：${out.length} 段，覆盖率 ${(cov.ratio * 100).toFixed(2)}%，缺失片段 ${JSON.stringify(cov.miss)}`,
    );
    expect(cov.ratio, `缺失 n-gram: ${JSON.stringify(cov.miss)}`).toBeGreaterThanOrEqual(0.98);
  });

  it('每个片段不得超过 chunkChars（重叠前缀除外）', () => {
    const out = chunkMarkdown(DOC_ADVERSARIAL, { chunkChars: 120, overlapChars: 0 });
    const over = out.filter((c) => c.text.length > 120);
    // 说明：短片段并入前一段（MIN_CHUNK_CHARS=60）会让片段超过 chunkChars，这是设计选择；
    // 这里只记录超出幅度，若超出过多（>2 倍）说明切分失效。
    console.log(`超出 chunkChars 的片段：${JSON.stringify(over.map((c) => c.text.length))}`);
    for (const c of over) expect(c.text.length).toBeLessThanOrEqual(240);
  });

  it('只有标题没有正文的小节：标题不得消失', () => {
    const out = chunkMarkdown(DOC_ADVERSARIAL);
    const heads = out.flatMap((c) => [c.heading, c.text]).join('\n');
    expect(heads).toContain('只有标题没有正文的小节');
    expect(heads).toContain('另一个只有标题的小节');
  });

  it('空输入 / 纯空白 → 不产出片段', () => {
    expect(chunkMarkdown('')).toEqual([]);
    expect(chunkMarkdown('   \n\n  ')).toEqual([]);
  });

  it('CRLF 文档不丢内容', () => {
    const md = ['# 标题', '', '第一段中文内容。', '', '第二段中文内容。'].join('\r\n');
    const out = chunkMarkdown(md);
    expect(norm(joinChunks(out))).toContain(norm('第二段中文内容。'));
  });

  it('代码块内的 # 注释不得被当成 Markdown 标题（否则标题路径被代码注释污染、代码行被改形）', () => {
    const md = ['# 项目文档', '## 1. 代码实现', '```python', '# 计算准确率', 'def acc(): pass', '```', '## 2. 数据', '正文。'].join('\n');
    const out = chunkMarkdown(md);
    const polluted = out.filter((c) => c.heading.includes('计算准确率'));
    console.log(
      '代码注释污染后的 heading：' +
        JSON.stringify(out.map((c) => ({ heading: c.heading, text: c.text.slice(0, 24) }))),
    );
    expect(polluted.map((c) => c.heading)).toEqual([]);
  });
});

// ============================================================================
// 3. 切块：问答对完整性
// ============================================================================

/** pair i 的问句带 QAQ{i}，答句带 A{i}_ 标记；片段含答标记却无问标记即被切散。 */
function qaDoc(labelQ: string, labelA: string, pairs: number, qPad = 4, aPad = 56): string {
  const lines = ['QATEST'];
  for (let i = 1; i <= pairs; i++) {
    lines.push(`${labelQ}QAQ${i}的${'问'.repeat(qPad)}？`);
    lines.push(`${labelA}A${i}_0${'答'.repeat(aPad)}。`);
  }
  return lines.join('\n');
}

describe('D. 切块破坏问答对', () => {
  it('常规问答（每对都装得进 chunkChars）→ 不得出现「只有答没有问」', () => {
    const doc = qaDoc('问：', '答：', 3);
    const out = chunkMarkdown(doc, { chunkChars: 200, overlapChars: 0 });
    console.log(`常规问答分组：${JSON.stringify(out.map((c) => stripMark(c.text).slice(0, 20)))}`);
    expect(qaViolations(out, 3)).toEqual([]);
  });

  it('单个问答对超过 chunkChars（长问题 + 长答案）→ 不得出现「只有答没有问」', () => {
    const q = `问：QAQ1${'问'.repeat(68)}？`;
    const a = `答：A1_0${'答'.repeat(62)}。`;
    const q2 = `问：QAQ2${'短'.repeat(30)}？`;
    const a2 = `答：A2_0${'行'.repeat(30)}。`;
    const doc = ['QATEST', q, a, q2, a2].join('\n');
    const out = chunkMarkdown(doc, { chunkChars: 80, overlapChars: 0 });
    const bad = qaViolations(out, 2);
    console.log(`超长问答对（chunkChars=80/overlap=0）：${JSON.stringify(out.map((c) => stripMark(c.text).slice(0, 22)))}`);
    // 同一份内容在默认重叠（120）下会被相邻尾部意外修好 —— 一并记录
    const outDefault = chunkMarkdown(doc, { chunkChars: 80 });
    console.log(`同样内容但 overlap=120：破损 ${qaViolations(outDefault, 2).length} 处`);
    expect(bad, bad.join('；')).toEqual([]);
  });

  it('默认参数下：单个答案很长（>700 字）时，后半段会脱离问题标签（真实语料形状）', () => {
    const q = '问：QAQ1你在 RAG 项目里具体怎么做的检索优化？';
    const a = '答：' + Array.from({ length: 16 }, (_, k) => `A1_${k}${'我做了召回与重排的联动优化，并逐条复核了指标口径。'.repeat(2)}`).join('');
    const doc = ['# 面试问答', q, a].join('\n');
    const out = chunkMarkdown(doc); // 默认 700 / 120
    const bad = qaViolations(out, 1);
    console.log(`默认 700/120、单对超长答案：共 ${out.length} 段，丢问题标签 ${bad.length} 处 → ${JSON.stringify(bad)}`);
    expect(bad, bad.join('；')).toEqual([]);
  });

  it('Markdown 粗体标签 **问**：/**答**： 必须与「问：」等价（对照实验）', () => {
    const plain = qaDoc('问：', '答：', 3);
    const bold = qaDoc('**问**：', '**答**：', 3);
    // chunkChars=200：plain 版每一对都能整组装进一个片段（对照组应为 0 破损）；
    // bold 版因为标签没被识别，走 splitBody 贪心打包，边界会落到答案上。
    const outPlain = chunkMarkdown(plain, { chunkChars: 200, overlapChars: 0 });
    const outBold = chunkMarkdown(bold, { chunkChars: 200, overlapChars: 0 });
    const badPlain = qaViolations(outPlain, 3);
    const badBold = qaViolations(outBold, 3);
    console.log(`plain 破损 ${badPlain.length} 处；bold 破损 ${badBold.length} 处`);
    console.log(`bold 分块：${JSON.stringify(outBold.map((c) => stripMark(c.text).slice(0, 26)))}`);
    expect(badPlain, 'plain 对照组不应破损').toEqual([]);
    expect(badBold, `bold 标签未被识别为问答起点：${badBold.join('；')}`).toEqual([]);
  });

  it('问答密集文档（默认参数）→ 记录有多少片段丢掉问题标签', () => {
    const doc = ['# 面试问答', ...Array.from({ length: 6 }, (_, i) => [
      `问：QAQ${i + 1}你在项目里具体负责什么？`,
      `答：A${i + 1}_0${'我负责检索与评测，包括数据清洗、指标设计、结果复核。'.repeat(3)}`,
    ])].flat().join('\n');
    const out = chunkMarkdown(doc);
    const bad = qaViolations(out, 6);
    console.log(`默认 700/120、6 组问答：共 ${out.length} 段，丢问题标签 ${bad.length} 处 → ${JSON.stringify(bad)}`);
    expect(out.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// 4. 检索：负信号与阈值
// ============================================================================

const dimOf = 16;
const basis = (i: number, d = dimOf): number[] => Array.from({ length: d }, (_, k) => (k === i ? 1 : 0));

describe('E. 检索负信号（防瞎编的地基）', () => {
  const store = {
    all: (): RagChunk[] =>
      Array.from({ length: 6 }, (_, i) =>
        mkChunk({ id: `c${i}`, index: i, text: `第${i}段：我负责 RAG 检索与排序模块的开发。`, vector: basis(i) }),
      ).concat(mkChunk({ id: 'filler', index: 6, text: '无关内容。', vector: basis(6) })),
  };

  it('强相关 → lowConfidence=false；完全无相关内容 → lowConfidence=true（两者不得同结果）', () => {
    const strong = search(store, '你负责的检索模块是怎么做的', basis(0), { topK: 3 });
    const none = search(store, '如何用 Rust 写一个分布式数据库的 WAL 恢复逻辑', basis(15), { topK: 3 });
    console.log(`强相关：topScore=${strong.topScore} lowConfidence=${strong.lowConfidence}；无相关内容：topScore=${none.topScore} lowConfidence=${none.lowConfidence}`);
    expect(strong.lowConfidence).toBe(false);
    expect(none.lowConfidence).toBe(true);
    expect(strong.topScore).toBeGreaterThan(none.topScore);
  });

  it('关键词满分也不能把低余弦抬成「高置信」（lowConfidence 用的是原始余弦）', () => {
    const s = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'kw', text: '海洋遥感 知识服务平台 多智能体 海洋遥感 知识服务平台', vector: basis(14) }),
      ],
    };
    const r = search(s, '海洋遥感知识服务平台', basis(15), { topK: 1 });
    console.log(`关键词全命中但余弦=0：evidence=${r.evidence.length} topScore=${r.topScore} lowConfidence=${r.lowConfidence} keywordBonus=${r.evidence[0]?.keywordBonus}`);
    expect(r.lowConfidence).toBe(true);
    // 修复后契约（Lead 2026-10-17）：topScore === 0 → **清空证据**。
    // 纯关键词噪声没有语义依据，留着会让调用方 `if (evidence.length)`
    // 把完全无关的片段当依据注入提示词。
    expect(r.evidence.length).toBe(0);
  });

  it('阈值边界是严格小于（cos == minScore 视为高置信）', () => {
    const s = {
      all: (): RagChunk[] => [mkChunk({ id: 'x', text: '内容。', vector: [0.45, Math.sqrt(1 - 0.45 ** 2)] })],
    };
    const r = search(s, '内容', [1, 0], { minScore: 0.45, topK: 1 });
    console.log(`cos=${r.topScore}（=minScore）→ lowConfidence=${r.lowConfidence}`);
    expect(r.topScore).toBeCloseTo(0.45, 6);
    expect(r.lowConfidence).toBe(false);
  });

  it('minScore=0 下 0 分片段不再被当作证据（默认路径已封住契约陷阱）', () => {
    const s = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'a', text: '甲', vector: basis(0) }),
        mkChunk({ id: 'b', text: '乙', vector: basis(1) }),
      ],
    };
    const r = search(s, '完全无关', basis(15), { minScore: 0, topK: 2 });
    console.log(`minScore=0：evidence=${r.evidence.length} topScore=${r.topScore} lowConfidence=${r.lowConfidence}`);
    // 修复后契约：topScore === 0 → 清空证据。因此"全是 0 分却算证据"的陷阱
    // 在默认路径上已经不存在；`lowConfidence` 仍为 false（cos 0 不小于 minScore 0），
    // 但**没有证据**，所以调用方不可能误用无关片段。
    expect(r.evidence).toHaveLength(0);
    expect(r.topScore).toBe(0);
    expect(r.lowConfidence).toBe(false);
    // 自测路径（debug: true）保留全部候选，供设置面板展示真实分数分布
    const dbg = search(s, '完全无关', basis(15), { minScore: 0, topK: 2, debug: true });
    expect(dbg.evidence).toHaveLength(2);
    expect(dbg.evidence.every((e) => e.score === 0)).toBe(true);
  });

  it('tokenize/keywordScore/exactHits 基本性质', () => {
    expect(tokenize('知识服务平台')).toEqual(['知识', '识服', '服务', '务平', '平台']);
    expect(keywordScore(tokenize('数据库索引'), '前端样式')).toBe(0);
    expect(exactHits('RRF 的 K 取多少？', '没有相关词')).toBe(0);
  });
});

// ============================================================================
// 5. 维度不匹配 / 混合维度
// ============================================================================

describe('F. 维度不匹配（应返回 0/空，不得抛异常或给假高分）', () => {
  it('cosine 维度不一致返回 0，不抛异常', () => {
    expect(cosine([1, 0, 0], [1, 0])).toBe(0);
    expect(cosine([], [1, 0])).toBe(0);
    expect(cosine([0, 0], [0, 0])).toBe(0);
  });

  it('查询向量维度与库中不一致：不抛异常、topScore=0、低置信', () => {
    const s = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'a', text: 'RAG 检索模块的实现细节。', vector: [1, 0, 0, 0] }),
        mkChunk({ id: 'b', text: '前端样式调整。', vector: [0, 1, 0, 0] }),
      ],
    };
    const q = Array.from({ length: 8 }, (_, i) => (i === 0 ? 1 : 0));
    const r = search(s, 'RAG 检索', q, { topK: 2 });
    console.log(`维度 8 vs 4：topScore=${r.topScore} scores=${JSON.stringify(r.evidence.map((e) => e.score))} lowConfidence=${r.lowConfidence}`);
    expect(r.topScore).toBe(0);
    expect(r.lowConfidence).toBe(true);
  });

  it('维度不匹配的片段会被直接跳过，不得靠关键词加分抢到第 1 名', () => {
    const s = {
      all: (): RagChunk[] => [
        mkChunk({ id: 'good', text: '前端界面用 Tailwind 写样式。', vector: [1, 0, 0, 0] }), // cos 0.3
        mkChunk({ id: 'bad-dim', text: '用 RAG、RRF、FTS5、BM25 做混合检索召回。', vector: [1, 0] }), // 维度不符 → 跳过
      ],
    };
    const q = [0.3, Math.sqrt(1 - 0.09), 0, 0];
    const r = search(s, 'RAG RRF FTS5 BM25 混合检索', q, { topK: 2 });
    console.log(`排序：${JSON.stringify(r.evidence.map((e) => ({ id: e.chunkId, score: e.score, bonus: e.keywordBonus })))} lowConfidence=${r.lowConfidence}`);
    // 修复后（Lead 2026-10-17）：维度不符的片段在打分前就被跳过，
    // 不再出现 "score=0 却靠关键词加分排第一" 的假证据。
    expect(r.evidence.map((e) => e.chunkId)).toEqual(['good']);
    expect(r.lowConfidence).toBe(true);
  });

  it('同一模型换 dimensions 后重建单份资料：库内维度必须保持一致', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mc-rag-dim-'));
    const store = new RagStore(join(dir, 'index.json'));
    const settings: EmbeddingSettings = { providerId: 'x', baseUrl: 'https://x/v1', model: 'text-embedding-v3', dim: 4 };
    const fake = (dim: number) => async (_s: EmbeddingSettings, _k: string, texts: string[]) => ({
      vectors: texts.map(() => Array.from({ length: dim }, (_, i) => (i === 0 ? 1 : 0))),
      dim,
      latencyMs: 0,
    });
    await indexDocument(store, fake(4), { slot: 'resume', source: 'a.md', text: '# 甲\n简历正文内容。', embedding: settings, apiKey: 'k' });
    expect(store.dimension).toBe(4);
    // 用户把 settings.dim 从 4 改成 3（模型名不变），再索引 JD：
    // 修复后（Lead 2026-10-17）：indexer **拒绝**维度和现有索引不一致的索引，
    // 并给出中文原因——"拒绝本次索引"是验证者认可的两种正确行为之一。
    await expect(
      indexDocument(store, fake(3), { slot: 'jd', source: 'b.md', text: '# 乙\nJD 正文内容。', embedding: settings, apiKey: 'k' }),
    ).rejects.toThrow(/维度/);
    const dims = store.all().map((c) => c.vector?.length ?? 0);
    console.log(`换维度后：store.dimension=${store.dimension} 各片段维度=${JSON.stringify(dims)} chunkCount=${store.size}`);
    rmSync(dir, { recursive: true, force: true });
    // 库内维度必须一致，且被拒绝的那次没有污染 store
    expect(new Set(dims).size, `库内存在多种维度：${JSON.stringify(dims)}`).toBe(1);
    expect(store.dimension).toBe(4);
  });
});

// ============================================================================
// 6. embedder：端点规范化
// ============================================================================

describe('G. embeddingsEndpoint 规范化', () => {
  const CASES: [string, string][] = [
    ['https://dashscope.aliyuncs.com/compatible-mode/v1', 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'],
    ['https://dashscope.aliyuncs.com/compatible-mode/v1/', 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'],
    ['https://api.openai.com/v1', 'https://api.openai.com/v1/embeddings'],
    ['https://api.openai.com', 'https://api.openai.com/v1/embeddings'],
    ['https://api.siliconflow.cn/v1', 'https://api.siliconflow.cn/v1/embeddings'],
    ['http://127.0.0.1:1234/v1', 'http://127.0.0.1:1234/v1/embeddings'],
    ['https://x/v1/embeddings', 'https://x/v1/embeddings'],
    ['https://x/v1/embeddings/', 'https://x/v1/embeddings'],
    // 用户从厂商文档里复制过来的完整 chat 端点（很常见）：
    ['https://api.openai.com/v1/chat/completions', 'https://api.openai.com/v1/embeddings'],
    ['https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'],
  ];

  it.each(CASES)('%s → %s', (input, want) => {
    expect(embeddingsEndpoint(input)).toBe(want);
  });

  it('空串 / 纯空白必须抛错（不得拼出垃圾 URL）', () => {
    expect(() => embeddingsEndpoint('')).toThrow();
    expect(() => embeddingsEndpoint('   ')).toThrow();
  });

  it('记录所有输入的规范化结果', () => {
    const rows = CASES.map(([i]) => `${i} → ${(() => { try { return embeddingsEndpoint(i); } catch (e) { return `抛错：${(e as Error).message}`; } })()}`);
    console.log('\n' + rows.join('\n'));
    expect(rows.length).toBe(CASES.length);
  });
});

// ============================================================================
// 7. embedder：请求形状与错误处理（mock fetch，不联网）
// ============================================================================

interface MockCall {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

function installFetch(
  handler: (call: MockCall) => { status: number; body: string } | Promise<{ status: number; body: string }>,
): MockCall[] {
  const calls: MockCall[] = [];
  const fn = async (url: unknown, init?: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {};
    const raw = init?.headers as Record<string, string> | undefined;
    if (raw) for (const [k, v] of Object.entries(raw)) headers[k.toLowerCase()] = String(v);
    const call: MockCall = {
      url: String(url),
      method: init?.method ?? 'GET',
      headers,
      body: JSON.parse(String(init?.body ?? 'null')) as Record<string, unknown>,
    };
    calls.push(call);
    if (init?.signal?.aborted) {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      throw err;
    }
    const r = await handler(call);
    return new Response(r.body, { status: r.status, headers: { 'content-type': 'application/json' } });
  };
  vi.stubGlobal('fetch', fn as unknown as typeof fetch);
  return calls;
}

function embBody(n: number, dim: number, startIndex = 0): string {
  return JSON.stringify({
    object: 'list',
    data: Array.from({ length: n }, (_, i) => ({
      object: 'embedding',
      index: startIndex + i,
      embedding: Array.from({ length: dim }, (_, d) => Number(((i + d + 1) / (dim + n)).toFixed(6))),
    })),
    usage: { total_tokens: n },
  });
}

const SETTINGS: EmbeddingSettings = {
  providerId: 'dashscope',
  baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  model: 'text-embedding-v3',
  dim: 4,
  batchSize: 2,
};

afterEach(() => vi.unstubAllGlobals());

describe('H. embedder 请求形状', () => {
  it('POST 到 /embeddings，带 Bearer 头、model/input/encoding_format/dimensions', async () => {
    const calls = installFetch(() => ({ status: 200, body: embBody(1, 4) }));
    await embedTexts(SETTINGS, 'sk-test', ['甲'], {});
    const c = calls[0];
    console.log(`请求：${c.method} ${c.url} body=${JSON.stringify(c.body)} auth=${c.headers.authorization}`);
    expect(c.method).toBe('POST');
    expect(c.url).toBe('https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings');
    expect(c.headers.authorization).toBe('Bearer sk-test');
    expect(c.body.model).toBe('text-embedding-v3');
    expect(c.body.input).toEqual(['甲']);
    expect(c.body.encoding_format).toBe('float');
    expect(c.body.dimensions).toBe(4);
  });

  it('dim 未设置/为 0 时不发送 dimensions（避免不支持的端点 400）', async () => {
    const calls = installFetch(() => ({ status: 200, body: embBody(1, 4) }));
    await embedTexts({ ...SETTINGS, dim: 0 }, 'sk-test', ['甲'], {});
    expect(calls[0].body.dimensions).toBeUndefined();
  });

  it('空输入不发请求；缺 key 直接抛错', async () => {
    const calls = installFetch(() => ({ status: 200, body: '{}' }));
    const r = await embedTexts(SETTINGS, 'sk-test', []);
    expect(calls).toHaveLength(0);
    expect(r.vectors).toEqual([]);
    await expect(embedTexts(SETTINGS, '', ['甲'])).rejects.toThrow(/API Key/);
    expect(calls).toHaveLength(0);
  });

  it('服务端乱序返回时按 index 归位', async () => {
    installFetch(() => ({
      status: 200,
      body: JSON.stringify({ data: [{ index: 1, embedding: [9, 9, 9, 9] }, { index: 0, embedding: [1, 1, 1, 1] }] }),
    }));
    const r = await embedTexts(SETTINGS, 'sk-test', ['甲', '乙'], {});
    expect(r.vectors[0]).toEqual([1, 1, 1, 1]);
    expect(r.vectors[1]).toEqual([9, 9, 9, 9]);
  });
});

describe('I. embedder 错误处理（401/403 不重试；429/5xx 重试；条数不符必须报错）', () => {
  it('401 立即失败，只发 1 次请求', async () => {
    const calls = installFetch(() => ({ status: 401, body: JSON.stringify({ error: { message: 'Incorrect API key provided.' } }) }));
    await expect(embedTexts(SETTINGS, 'sk-bad', ['甲'], {})).rejects.toThrow(/HTTP 401/);
    console.log(`401 请求次数=${calls.length}`);
    expect(calls).toHaveLength(1);
  });

  it('403 立即失败，只发 1 次请求', async () => {
    const calls = installFetch(() => ({ status: 403, body: JSON.stringify({ error: { message: 'Forbidden' } }) }));
    await expect(embedTexts(SETTINGS, 'sk-bad', ['甲'], {})).rejects.toThrow(/HTTP 403/);
    expect(calls).toHaveLength(1);
  });

  it('429 会重试（3 次后抛出）', async () => {
    const calls = installFetch(() => ({ status: 429, body: JSON.stringify({ error: { message: 'rate limited' } }) }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲'], {})).rejects.toThrow(/HTTP 429/);
    console.log(`429 请求次数=${calls.length}`);
    expect(calls.length).toBeGreaterThan(1);
  });

  it('500 会重试（3 次后抛出）', async () => {
    const calls = installFetch(() => ({ status: 500, body: 'upstream boom' }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲'], {})).rejects.toThrow(/HTTP 500/);
    expect(calls.length).toBeGreaterThan(1);
  });

  it('200 但返回条数少于请求条数 → 必须抛错，且不得重试/降半批（不得静默留洞）', async () => {
    // 服务端永远只回 1 条（模拟"按条计费/截断/网关丢包"）。
    // 注意：00:35 版实现会先重试 3 次再降半批，最终静默返回 [v0, <hole>]；
    // 00:41 版实现改成 ResponseShapeError 立即抛出 —— 本用例锁的就是这个行为。
    const calls = installFetch(() => ({ status: 200, body: embBody(1, 4) }));
    await expect(embedTexts({ ...SETTINGS, batchSize: 2 }, 'sk-test', ['甲', '乙'], {})).rejects.toThrow(/返回条数不符/);
    console.log(`条数不符：请求次数=${calls.length}（期望 1：形状错误不重试）`);
    expect(calls).toHaveLength(1);
  });

  it('200 但返回条数多于请求条数 → 抛错（不得错位）', async () => {
    installFetch(() => ({ status: 200, body: embBody(3, 4) }));
    await expect(embedTexts({ ...SETTINGS, batchSize: 2 }, 'sk-test', ['甲', '乙'], {})).rejects.toThrow(/返回条数不符/);
  });

  it('200 但 index 越界 → 抛错（不得把向量写到别的槽位）', async () => {
    installFetch(() => ({
      status: 200,
      body: JSON.stringify({ data: [{ index: 0, embedding: [1, 1, 1, 1] }, { index: 5, embedding: [2, 2, 2, 2] }] }),
    }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲', '乙'], {})).rejects.toThrow(/越界/);
  });

  it('200 但 index 重复 → 抛错', async () => {
    installFetch(() => ({
      status: 200,
      body: JSON.stringify({ data: [{ index: 0, embedding: [1, 1, 1, 1] }, { index: 0, embedding: [2, 2, 2, 2] }] }),
    }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲', '乙'], {})).rejects.toThrow(/重复/);
  });

  it('端点不支持 dimensions（400）时自动去掉该参数重试一次', async () => {
    const calls = installFetch((c) =>
      c.body.dimensions ? { status: 400, body: JSON.stringify({ error: { message: 'dimensions not supported' } }) } : { status: 200, body: embBody(1, 8) },
    );
    const r = await embedTexts(SETTINGS, 'sk-test', ['甲'], {});
    console.log(`400 降级：调用 ${calls.length} 次；第 2 次 body=${JSON.stringify(calls[1]?.body)}；返回维度=${r.dim}（请求的是 4）`);
    expect(calls).toHaveLength(2);
    expect(calls[1].body.dimensions).toBeUndefined();
    expect(r.dim).toBe(8);
    // 记录性：服务端无视 dimensions 返回 8 维时，embedTexts 不校验「返回维度==请求维度」，
    // 库里的维度会与 settings.dim 不一致（见 F 组"库内维度必须保持一致"）
  });

  it('200 但不是 JSON → 可读错误', async () => {
    installFetch(() => ({ status: 200, body: '<html>gateway</html>' }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲'], {})).rejects.toThrow(/不是 JSON/);
  });

  it('200 但返回空向量 → 抛错', async () => {
    installFetch(() => ({ status: 200, body: JSON.stringify({ data: [{ index: 0, embedding: [] }] }) }));
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲'], {})).rejects.toThrow(/空向量/);
  });

  it('已取消的请求必须立即停止，不再重试', async () => {
    const ac = new AbortController();
    ac.abort();
    const calls = installFetch(() => ({ status: 200, body: embBody(1, 4) }));
    const t0 = Date.now();
    await expect(embedTexts(SETTINGS, 'sk-test', ['甲'], { signal: ac.signal })).rejects.toThrow();
    const dt = Date.now() - t0;
    console.log(`取消后：请求次数=${calls.length} 耗时=${dt}ms`);
    // 修复后（Lead 2026-10-17）：signal 已 aborted 时**一个请求都不发**，
    // 直接抛原生 AbortError。原断言为 1（修复前会先发一次再被中断）。
    expect(calls).toHaveLength(0);
    expect(dt).toBeLessThan(300);
  });

  it('取消一个整批索引任务：应立刻停止，不得继续重试+降半批', async () => {
    const ac = new AbortController();
    ac.abort();
    const calls = installFetch(() => ({ status: 200, body: embBody(1, 4) }));
    const texts = Array.from({ length: 16 }, (_, i) => `第${i}条`);
    const t0 = Date.now();
    await expect(embedTexts({ ...SETTINGS, batchSize: 16 }, 'sk-test', texts, { signal: ac.signal })).rejects.toThrow();
    const dt = Date.now() - t0;
    console.log(`取消 16 条批次：请求次数=${calls.length} 耗时=${dt}ms`);
    expect(calls.length, '取消后不该继续发请求').toBeLessThanOrEqual(1);
  });
});

// ============================================================================
// 8. store 落盘与加载
// ============================================================================

describe('J. store 落盘 / 反序列化', () => {
  const dir = mkdtempSync(join(tmpdir(), 'mc-rag-store-'));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it('persist 后重新加载内容一致', () => {
    const file = join(dir, 'index.json');
    const s = new RagStore(file);
    s.replaceSlot('resume', 'a.md', [{ text: '甲', heading: 'H', index: 0 }], ['claimed'], [[1, 2, 3]], 'text-embedding-v3');
    s.persist();
    const s2 = new RagStore(file);
    expect(s2.size).toBe(1);
    expect(s2.modelId).toBe('text-embedding-v3');
    expect(s2.dimension).toBe(3);
    expect(s2.all()[0].factLevel).toBe('claimed');
    expect(s2.all()[0].vector).toEqual([1, 2, 3]);
  });

  it('索引文件损坏 / 版本不符 → 退化为空索引，不抛异常', () => {
    const bad = join(dir, 'bad.json');
    writeFileSync(bad, '{ not json', 'utf8');
    expect(new RagStore(bad).size).toBe(0);
    const ver = join(dir, 'ver.json');
    writeFileSync(ver, JSON.stringify({ version: 2, chunks: [{ id: 'x' }] }), 'utf8');
    expect(new RagStore(ver).size).toBe(0);
  });

  it('clear 后落盘为空索引', () => {
    const file = join(dir, 'clear.json');
    const s = new RagStore(file);
    s.replaceSlot('resume', 'a.md', [{ text: '甲', heading: '', index: 0 }], ['verified'], [[1]], 'm');
    s.persist();
    s.clear();
    expect(s.size).toBe(0);
    expect(JSON.parse(readFileSync(file, 'utf8')).chunks).toEqual([]);
  });
});

// ============================================================================
// 9. indexer 集成
// ============================================================================

describe('K. indexer 集成', () => {
  it('事实档位随片段入库，且被标注为 claimed 的片段能完整取回', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mc-rag-idx-'));
    const store = new RagStore(join(dir, 'index.json'));
    const text = '# 项目经历\n\n我在项目里把准确率从 78% 提到了 91.3%。';
    const r = await indexDocument(
      store,
      async (_s, _k, texts) => ({ vectors: texts.map(() => [1, 0, 0, 0]), dim: 4, latencyMs: 1 }),
      { slot: 'resume', source: 'a.md', text, embedding: SETTINGS, apiKey: 'k' },
    );
    console.log(`indexDocument：chunks=${r.chunks} byLevel=${JSON.stringify(r.byLevel)} factLevel=${store.all()[0].factLevel}`);
    expect(r.byLevel.claimed).toBe(1);
    expect(store.all()[0].factLevel).toBe('claimed');
    expect(store.all()[0].text).toContain('91.3%');
    rmSync(dir, { recursive: true, force: true });
  });

  it('上游 embedder 静默留洞（稀疏数组）→ indexer 必须整批失败，不得静默入库半截', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mc-rag-hole-'));
    const store = new RagStore(join(dir, 'index.json'));
    // 两个段落各 180 字、chunkChars=200 → 稳定产出 2 个片段（避免 MIN_CHUNK_CHARS=60 的合并把片段并成 1 个）
    const text = ['# 甲', '', '第一段内容。'.repeat(30), '', '第二段内容。'.repeat(30)].join('\n');
    // 修复后（Lead 2026-10-17）：indexer 自检"每个向量非空"，留洞时**抛出**并中止入库。
    await expect(
      indexDocument(
        store,
        async (_s, _k, texts) => {
          const vectors: number[][] = new Array(texts.length);
          vectors[0] = [1, 0, 0, 0]; // 第 2 条缺失 —— 上游 embedder 若静默留洞就是这个形状
          return { vectors, dim: 4, latencyMs: 1 };
        },
        { slot: 'resume', source: 'a.md', text, embedding: SETTINGS, apiKey: 'k', chunk: { chunkChars: 200, overlapChars: 0 } },
      ),
    ).rejects.toThrow(/向量缺失/);
    const noVec = store.all().filter((c) => !c.vector || c.vector.length === 0);
    console.log(`留洞后入库片段=${store.size} 无向量片段=${noVec.length}`);
    rmSync(dir, { recursive: true, force: true });
    // 入库的片段必须都有向量：留洞时整批失败，store 里不留半截
    expect(noVec.length, `无向量的片段永远不会被检索到：${JSON.stringify(noVec.map((c) => c.id))}`).toBe(0);
    expect(store.size).toBe(0);
  });
});
