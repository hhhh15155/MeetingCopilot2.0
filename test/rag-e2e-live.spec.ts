/**
 * 真实 embedding 端到端校准（需要真 API Key，默认跳过）。
 *
 * 为什么必须有这个脚本：`search.ts` 的 `DEFAULT_MIN_SCORE = 0.45` 是初始估值，
 * **没有实测依据**。这个阈值决定了"什么算资料里没找到"，直接关系到会不会
 * 让模型在无依据时硬编——所以必须用自己的真实语料校准，不能靠猜。
 *
 * 用法（PowerShell）：
 *   $env:DASHSCOPE_API_KEY='sk-xxx'; npx vitest run test/rag-e2e-live.spec.ts
 * 可选：
 *   $env:RAG_E2E_MODEL='text-embedding-v3'
 *   $env:RAG_E2E_BASEURL='https://dashscope.aliyuncs.com/compatible-mode/v1'
 *   $env:RAG_CORPUS='D:\path\to\你的项目讲稿.md'   # 用你真实资料跑，校准更准
 *
 * 它会输出：
 *   * 真实 embedding 是否可用、维度、耗时
 *   * 每轮索引耗时与费用量级
 *   * **分数分布**：强相关问题 vs 无关问题的原始余弦，据此给出建议阈值
 */
import { existsSync, readFileSync, mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { describe, expect, it } from 'vitest';
import { embedTexts } from '../electron/rag/embedder';
import { classifyChunk } from '../electron/rag/factlevel';
import { indexDocument } from '../electron/rag/indexer';
import { search } from '../electron/rag/search';
import { RagStore } from '../electron/rag/store';

const API_KEY = process.env.DASHSCOPE_API_KEY ?? process.env.RAG_E2E_API_KEY ?? '';
const BASE_URL = process.env.RAG_E2E_BASEURL ?? 'https://dashscope.aliyuncs.com/compatible-mode/v1';
const MODEL = process.env.RAG_E2E_MODEL ?? 'text-embedding-v3';
const CORPUS_PATH = process.env.RAG_CORPUS ?? '';
const live = API_KEY ? describe : describe.skip;

/**
 * 内置样例语料：刻意混入三种事实档位，用来验证
 * ① 标注是否生效 ② 强相关/无关问题的分数是否分得开。
 * 若设置了 RAG_CORPUS，则用真实资料替换。
 */
const SAMPLE_CORPUS = `# 面试问答速查

## 第 1 部分 项目概述

问：介绍一下你的海洋遥感多智能体知识服务平台。
答：我做的是一个面向海洋遥感领域的多智能体知识服务平台。前端用 React，后端用 Python，检索部分我自己实现了向量加 BM25 的混合召回，用 RRF 做融合，K 取 60。这部分代码在 marine-knowledge-platform 仓库的 retrieval 模块里，跑通过。

问：你的 RAG 召回率是多少？
答：简历上写的是 88.7%，测试集 500 条。不过我要如实说明，这个数字在仓库里没有对应的数据集和可复算的报告支撑。

## 第 2 部分 技术细节

问：为什么要用 RRF 而不是简单加权？
答：因为向量检索和关键词检索的分数尺度不可比，直接加权会被某一侧的分布带偏。RRF 只用排名，天然规避了尺度问题。

问：后续有什么优化计划？
答：预计接入 vLLM 做推理量化，吞吐大概能提升 3 倍左右，这块还在规划中。

## 第 3 部分 边界

问：你们做过 Docker 部署和 CI 吗？
答：这部分我没有实际做过，项目是在本地环境跑的。如果要做，我的思路是先用多阶段构建把镜像压小，再用 GitHub Actions 串起来。
`;

function sampleChunks(text: string): { verified: number; claimed: number; unverified: number } {
  // 粗略按段落看档位分布（仅用于日志展示）
  const out = { verified: 0, claimed: 0, unverified: 0 };
  for (const p of text.split(/\n{2,}/)) {
    if (!p.trim()) continue;
    out[classifyChunk(p)]++;
  }
  return out;
}

live('真实 embedding 端到端校准', () => {
  it(
    '索引 + 检索，并输出分数分布以校准 minScore',
    async () => {
      const corpus = CORPUS_PATH && existsSync(CORPUS_PATH) ? readFileSync(CORPUS_PATH, 'utf8') : SAMPLE_CORPUS;
      console.log(`\n语料：${CORPUS_PATH && existsSync(CORPUS_PATH) ? CORPUS_PATH : '（内置样例）'}，${corpus.length} 字`);
      console.log('档位分布（按段落粗算）：', sampleChunks(corpus));

      const dir = mkdtempSync(join(tmpdir(), 'rag-e2e-'));
      const store = new RagStore(join(dir, 'index.json'));
      const settings = { providerId: 'e2e', baseUrl: BASE_URL, model: MODEL, dim: 1024, batchSize: 16 };

      // 1) 连通性
      const t0 = Date.now();
      const probe = await embedTexts({ ...settings, batchSize: 1 }, API_KEY, ['连通性测试']);
      console.log(`embedding 可用：维度 ${probe.dim}，单条 ${Date.now() - t0}ms`);

      // 2) 建立索引
      const t1 = Date.now();
      const res = await indexDocument(store, embedTexts, {
        slot: 'resume',
        source: '面试问答速查.md',
        text: corpus,
        embedding: settings,
        apiKey: API_KEY,
      });
      console.log(`索引进度：${res.chunks} 片段，维度 ${res.dim}，耗时 ${Date.now() - t1}ms`);
      console.log('片段档位分布：', res.byLevel);

      // 3) 分数分布：强相关问题应当明显高于无关问题
      const strong = [
        '你的 RAG 召回率是多少？',
        '为什么要用 RRF 融合？',
        '你做过 Docker 部署吗？',
        '介绍一下你的项目',
      ];
      const weak = [
        '今天天气怎么样',
        '你会做红烧肉吗',
        '请介绍一下量子纠缠的数学基础',
      ];

      const qvec = async (q: string) => (await embedTexts({ ...settings, batchSize: 1 }, API_KEY, [q])).vectors[0];

      const rows: { kind: string; q: string; top: number; hit: string }[] = [];
      for (const q of strong) {
        const v = await qvec(q);
        const r = search(store, q, v, { topK: 3, minScore: 0, neighborExpand: false });
        rows.push({ kind: '相关', q, top: r.topScore, hit: r.evidence[0]?.heading ?? '' });
      }
      for (const q of weak) {
        const v = await qvec(q);
        const r = search(store, q, v, { topK: 3, minScore: 0, neighborExpand: false });
        rows.push({ kind: '无关', q, top: r.topScore, hit: r.evidence[0]?.heading ?? '' });
      }

      console.log('\n=== 分数分布（原始余弦）===');
      for (const r of rows) console.log(`[${r.kind}] ${r.top.toFixed(4)}  ${r.q}   → ${r.hit}`);

      const strongTops = rows.filter((r) => r.kind === '相关').map((r) => r.top);
      const weakTops = rows.filter((r) => r.kind === '无关').map((r) => r.top);
      const minStrong = Math.min(...strongTops);
      const maxWeak = Math.max(...weakTops);
      console.log(`\n相关问题最低分：${minStrong.toFixed(4)}`);
      console.log(`无关问题最高分：${maxWeak.toFixed(4)}`);
      if (minStrong > maxWeak) {
        const suggested = (minStrong + maxWeak) / 2;
        console.log(`✅ 可分：建议 minScore ≈ ${suggested.toFixed(2)}（取两者中点）`);
      } else {
        console.log(
          `⚠️ 不可分：无关问题得分(${maxWeak.toFixed(4)}) ≥ 相关问题最低分(${minStrong.toFixed(4)})，` +
            `单一阈值无法区分。建议：降低 minScore 到 ${(maxWeak + 0.02).toFixed(2)} 并把判断交给引用约束，` +
            `或提高切块粒度/更换 embedding 模型。`,
        );
      }

      // 4) 把校准结果落成文件，便于 Lead 读取（避免只靠控制台输出）
      const report = {
        corpus: CORPUS_PATH || '(内置样例)',
        chars: corpus.length,
        model: MODEL,
        dim: res.dim,
        chunks: res.chunks,
        byLevel: res.byLevel,
        rows,
        minStrong,
        maxWeak,
        separable: minStrong > maxWeak,
        suggestedMinScore: minStrong > maxWeak ? Number(((minStrong + maxWeak) / 2).toFixed(2)) : Number((maxWeak + 0.02).toFixed(2)),
      };
      const outPath = join(process.cwd(), 'rag-calibration.json');
      const { writeFileSync } = await import('fs');
      writeFileSync(outPath, JSON.stringify(report, null, 2), 'utf8');
      console.log(`校准报告已写入：${outPath}`);

      rmSync(dir, { recursive: true, force: true });

      expect(res.chunks).toBeGreaterThan(0);
      expect(res.dim).toBeGreaterThan(0);
    },
    180_000,
  );
});
