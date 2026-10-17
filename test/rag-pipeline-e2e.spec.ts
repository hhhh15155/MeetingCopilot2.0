/**
 * 端到端管线集成测试（不需要 API Key）。
 *
 * 用一份"真实语料形状"的求职资料，把 RAG 的**完整链路**串起来验证：
 *
 *   切块 → 事实档位标注 → 向量入库 → 检索 → 证据注入 prompt → 引用覆盖率审计
 *
 * 为什么这个测试值得存在：单测各自覆盖一个模块，但"防瞎编"是**跨模块**的
 * 性质——档位标对了却注入错了、证据注入了却没进最终 messages、引用了不存在的
 * 编号却没被发现，任何一环断了，单测仍然全绿而功能实际失效。
 *
 * 向量用确定性的"词重叠"假向量（同词越多余弦越高），因此可以不联网地
 * 真实检验检索排序与低置信判定，而不是把结果写死。
 */
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { describe, expect, it } from 'vitest';
import { chunkMarkdown } from '../electron/rag/chunker';
import { indexDocument } from '../electron/rag/indexer';
import { search, tokenize } from '../electron/rag/search';
import { RagStore } from '../electron/rag/store';
import { buildAnswerMessages, buildFallbackAnswer } from '../electron/llm/prompts';
import { auditCitations } from '../shared/citationAudit';
import type { EmbeddingSettings } from '../electron/rag/embedder';
import type { RagEvidence } from '../shared/protocol';

/** 真实语料形状：带【项目】/【边界】人工标注、含包装数字与规划口吻 */
const CORPUS = `# 面试问答速查

## 第 1 部分 项目概述

问：介绍一下你的海洋遥感多智能体知识服务平台。
答：我做的是一个面向海洋遥感领域的多智能体知识服务平台。前端用 React，后端用 Python，检索部分我自己实现了向量加 BM25 的混合召回，用 RRF 做融合，K 取 60。这部分代码在 marine-knowledge-platform 仓库的 retrieval 模块里，实测跑通过。

问：你的 RAG 召回率是多少？
答：简历上写的是 88.7%，测试集 500 条。我要如实说明，这个数字在仓库里没有对应的数据集和可复算的报告支撑。

## 第 2 部分 技术细节

问：为什么要用 RRF 而不是简单加权？
答：因为向量检索和关键词检索的分数尺度不可比，直接加权会被某一侧的分布带偏。RRF 只用排名，天然规避了尺度问题。

问：后续有什么优化计划？
答：预计接入 vLLM 做推理量化，吞吐大概能提升 3 倍左右，这块还在规划中。

## 第 3 部分 边界

问：你们做过 Docker 部署和 CI 吗？
答：这部分我没有实际做过，项目是在本地环境跑的。如果要做，我的思路是先用多阶段构建把镜像压小，再用 GitHub Actions 串起来。
`;

const DIM = 96;
const SETTINGS: EmbeddingSettings = { providerId: 'test', baseUrl: 'https://x/v1', model: 'fake', dim: DIM };

/** 确定性假向量：中文 bigram 词元哈希到维度，同词多则余弦高 */
function fakeVec(text: string): number[] {
  const v = new Array(DIM).fill(0);
  for (const tok of tokenize(text)) {
    let h = 0;
    for (const ch of tok) h = (h * 31 + ch.charCodeAt(0)) % 100003;
    v[h % DIM] += 1;
  }
  const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
  return v.map((x) => x / n);
}

async function buildIndex() {
  const dir = mkdtempSync(join(tmpdir(), 'rag-e2e-'));
  const store = new RagStore(join(dir, 'index.json'));
  await indexDocument(
    store,
    async (_s, _k, texts) => ({ vectors: texts.map(fakeVec), dim: DIM, latencyMs: 0 }),
    { slot: 'resume', source: '面试问答速查.md', text: CORPUS, embedding: SETTINGS, apiKey: 'k' },
  );
  return { store, dir };
}

/**
 * `ChatMessage.content` 是联合类型（文本 / 多模态分片），直接当字符串用会类型报错。
 * 取纯文本内容：字符串原样，数组则拼接其中的 text 分片。
 */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (typeof p === 'string' ? p : ((p as { text?: string }).text ?? '')))
      .join('\n');
  }
  return '';
}

describe('端到端：检索 → 注入 → 审计', () => {
  it('检索能命中对应片段，并保留事实档位', async () => {
    const { store, dir } = await buildIndex();
    const r = search(store, '你的 RAG 召回率是多少', fakeVec('你的 RAG 召回率是多少'), {
      topK: 3,
      minScore: 0,
      debug: true,
    });
    expect(r.evidence.length).toBeGreaterThan(0);
    // 召回率那段含包装数字（500 条 / 88.7%）→ 必须是 claimed
    const hit = r.evidence.find((e) => e.text.includes('88.7%'));
    expect(hit, '应命中含 88.7% 的片段').toBeDefined();
    expect(hit!.factLevel).toBe('claimed');
    rmSync(dir, { recursive: true, force: true });
  });

  it('规划口吻的片段被标成 unverified（不能当已完成讲）', async () => {
    const { store, dir } = await buildIndex();
    const r = search(store, '后续有什么优化计划', fakeVec('后续有什么优化计划'), {
      topK: 3,
      minScore: 0,
      debug: true,
    });
    const hit = r.evidence.find((e) => e.text.includes('vLLM'));
    expect(hit, '应命中含 vLLM 的片段').toBeDefined();
    expect(hit!.factLevel).toBe('unverified');
    rmSync(dir, { recursive: true, force: true });
  });

  it('无关问题低于阈值 → lowConfidence（主进程据此短路，不走模型）', async () => {
    const { store, dir } = await buildIndex();
    // 用生产阈值（不是 debug 的 0）
    const r = search(store, '你会做红烧肉吗', fakeVec('你会做红烧肉吗'), { topK: 5 });
    expect(r.lowConfidence).toBe(true);
    // 证据契约：分数在 (0, minScore) 之间时会**保留**证据（"桌上有什么料"是真实信息，
    // 界面要显示分数），但 lowConfidence 已置位 —— 由主进程短路成兜底话术，
    // 不把弱证据交给模型自由发挥。只有 topScore === 0 才会清空证据。
    if (r.evidence.length > 0) {
      expect(r.topScore).toBeLessThan(0.45);
      expect(r.topScore).toBeGreaterThan(0);
    }
    // 关键性质：无关问题的最高分必须明显低于相关问题（否则阈值无意义）
    const relevant = search(store, '你的 RAG 召回率是多少', fakeVec('你的 RAG 召回率是多少'), { topK: 5 });
    expect(r.topScore).toBeLessThan(relevant.topScore);
    rmSync(dir, { recursive: true, force: true });
  });

  it('证据真的进了最终 messages，且位置在 memo 之后', async () => {
    const { store, dir } = await buildIndex();
    const r = search(store, '你的 RAG 召回率是多少', fakeVec('你的 RAG 召回率是多少'), {
      topK: 3,
      minScore: 0,
      debug: true,
    });
    const msgs = buildAnswerMessages({
      mode: 'continuous',
      question: '你的 RAG 召回率是多少？',
      recentTranscript: ['面试官：你的 RAG 召回率是多少？'],
      resume: '',
      memo: '【已问问题】\n1. 自我介绍',
      evidence: r.evidence,
      requireCitations: true,
    });
    const joined = msgs.map((m) => textOf(m.content)).join('\n---\n');
    // 证据正文必须在最终请求里
    expect(joined).toContain('88.7%');
    // 档位规则与硬性约束必须在
    expect(joined).toContain('目标值/待确认');
    expect(joined).toContain('禁止换算、取整、外推');
    expect(joined).toContain('指不到来源的事实句，一句都不要写出来');
    // 必须点明"这些片段就是我的真实资料"，否则模型会因 persona 而不敢用
    expect(joined).toContain('上面这些片段就是我的真实资料');
    // 顺序：memo 在证据之前
    const iMemo = msgs.findIndex((m) => textOf(m.content).includes('【面试进行备忘】'));
    const iEv = msgs.findIndex((m) => textOf(m.content).includes('【可用资料片段】'));
    expect(iMemo).toBeGreaterThanOrEqual(0);
    expect(iEv).toBeGreaterThan(iMemo);
    // 最后一条 user 消息才是本轮提问
    expect(msgs[msgs.length - 1].role).toBe('user');
    expect(textOf(msgs[msgs.length - 1].content)).toContain('面试官刚才说');
    rmSync(dir, { recursive: true, force: true });
  });

  it('无证据时 messages 与老路径一致（不因 RAG 改动而多出内容）', async () => {
    const base = {
      mode: 'continuous' as const,
      question: '介绍一下你自己',
      recentTranscript: ['面试官：介绍一下你自己'],
      resume: '我做过检索优化。',
    };
    const withoutKey = buildAnswerMessages(base);
    const withEmpty = buildAnswerMessages({ ...base, evidence: [] });
    expect(withEmpty).toEqual(withoutKey);
  });

  it('用真实证据审计模型回答：抓出没有出处的事实句', async () => {
    const { store, dir } = await buildIndex();
    const r = search(store, '你的 RAG 召回率是多少', fakeVec('你的 RAG 召回率是多少'), {
      topK: 3,
      minScore: 0,
      debug: true,
    });
    const evidence: RagEvidence[] = r.evidence;
    // 模拟一个"半接地气"的回答：有依据的部分带引用，另有一句凭空多出的数字
    const answer = [
      `简历上写的是 88.7%，测试集 500 条，这个数字在仓库里没有可复算的报告支撑 [${evidence[0].ref}]。`,
      '另外我把推理延迟压到了 120ms。',
    ].join('\n');
    const audit = auditCitations(answer, evidence);
    expect(audit.cited).toEqual([evidence[0].ref]);
    expect(audit.uncitedFacts).toHaveLength(1);
    expect(audit.uncitedFacts[0]).toContain('120ms');
    expect(audit.coverage).toBeCloseTo(0.5, 5);
  });

  it('低置信时走兜底话术：是可直接念的中文，且不暴露 AI 存在', () => {
    for (const reason of ['no_index', 'low_score', 'no_evidence'] as const) {
      const a = buildFallbackAnswer(reason);
      expect(a.length, reason).toBeGreaterThan(10);
      expect(a, reason).not.toMatch(/知识库|检索|索引|模型|AI/i);
      expect(a, reason).not.toMatch(/^#|\*\*/);
    }
  });
});
