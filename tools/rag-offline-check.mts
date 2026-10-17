/**
 * RAG 离线验证脚本（无需 API Key）：用真实语料形状检验
 * 切块 → 事实档位标注 → （假）向量 → 入库 → 检索 的完整链路，
 * 并打印每个片段的事实档位分布，便于人工核对"包装数字有没有被标出来"。
 *
 * 用法：npx vite-node tools/rag-offline-check.mts [语料路径]
 * 不传路径则用内置样例。
 */
import { readFileSync, existsSync, mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { chunkMarkdown } from '../electron/rag/chunker';
import { classifyChunk, classifySentence, FACT_LEVEL_LABEL } from '../electron/rag/factlevel';
import { indexDocument } from '../electron/rag/indexer';
import { search, tokenize, keywordScore } from '../electron/rag/search';
import { RagStore } from '../electron/rag/store';
import type { EmbeddingSettings } from '../electron/rag/embedder';

const SAMPLE = `# 面试问答速查

## 第 1 部分 项目概述

问：介绍一下你的海洋遥感多智能体知识服务平台。
答：我做的是一个面向海洋遥感领域的多智能体知识服务平台。前端用 React，后端用 Python，检索部分我自己实现了向量加 BM25 的混合召回，用 RRF 做融合，K 取 60。这部分代码在 marine-knowledge-platform 仓库的 retrieval 模块里，跑通过。

问：你的 RAG 召回率是多少？
答：简历上写的是 88.7%，测试集 500 条。不过我要如实说明，这个数字在仓库里没有对应的数据集和可复算的报告支撑。

## 第 2 部分 技术细节

\`\`\`python
# 计算准确率
def acc(pred, label):
    correct = sum(1 for p, l in zip(pred, label) if p == l)
    return correct / len(label)
\`\`\`

问：为什么要用 RRF 而不是简单加权？
答：因为向量检索和关键词检索的分数尺度不可比，直接加权会被某一侧的分布带偏。RRF 只用排名，天然规避了尺度问题。

问：后续有什么优化计划？
答：预计接入 vLLM 做推理量化，吞吐大概能提升 3 倍左右，这块还在规划中。

## 第 3 部分 边界

问：你们做过 Docker 部署和 CI 吗？
答：这部分我没有实际做过，项目是在本地环境跑的。如果要做，我的思路是先用多阶段构建把镜像压小，再用 GitHub Actions 串起来。
`;

const corpusPath = process.argv[2];
const corpus = corpusPath && existsSync(corpusPath) ? readFileSync(corpusPath, 'utf8') : SAMPLE;
console.log(`语料：${corpusPath && existsSync(corpusPath) ? corpusPath : '（内置样例）'}  ${corpus.length} 字\n`);

// 1) 切块 + 档位
const drafts = chunkMarkdown(corpus);
console.log(`=== 切块（${drafts.length} 段）与事实档位 ===`);
let claimed = 0;
for (const d of drafts) {
  const lv = classifyChunk(`${d.heading}\n${d.text}`);
  if (lv === 'claimed') claimed++;
  const flag = lv === 'claimed' ? '★' : ' ';
  console.log(`${flag}[${d.index}] ${FACT_LEVEL_LABEL[lv].padEnd(6)} ${d.heading || '(无标题)'}`);
  console.log(`      ${JSON.stringify(d.text.slice(0, 76))}`);
}
console.log(`\n→ ${drafts.length} 段中 ${claimed} 段被标为「目标值/待确认」（需要模型谨慎处理）`);

// 2) 高危句逐句核对
console.log('\n=== 漏判自查（这些句子必须全部是 claimed）===');
const probes = [
  '我负责的模块准确率 91.3%。',
  'Macro-F1 达到 0.89。',
  '我把 QPS 从 300 提到了 1200。',
  'baseline 准确率 81.8。',
  '人类标注样本上的准确率是 92.4%。',
  '【项目】我负责的检索模块，准确率 91.3%。',
  '简历上写的是 88.7%，测试集 500 条。',
];
let leaks = 0;
for (const s of probes) {
  const lv = classifySentence(s);
  if (lv !== 'claimed') leaks++;
  console.log(`${lv === 'claimed' ? '✓' : '✗ 漏判'}  ${lv.padEnd(10)} ${s}`);
}
console.log(leaks === 0 ? '\n✅ 无漏判' : `\n❌ ${leaks} 处漏判`);

// 3) 假向量入库 + 检索（验证管线与中文关键词，不依赖网络）
const dim = 64;
/** 把文本映射成稳定的假向量：同词元多的文本余弦更高，足以验证管线连通性 */
const fakeVec = (t: string): number[] => {
  const v = new Array(dim).fill(0);
  for (const tok of tokenize(t)) {
    let h = 0;
    for (const ch of tok) h = (h * 31 + ch.charCodeAt(0)) % 100003;
    v[h % dim] += 1;
  }
  const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
  return v.map((x) => x / n);
};
const settings: EmbeddingSettings = { providerId: 'offline', baseUrl: 'https://x/v1', model: 'fake', dim };

const dir = mkdtempSync(join(tmpdir(), 'rag-offline-'));
const store = new RagStore(join(dir, 'index.json'));
const result = await indexDocument(store, async (_s, _k, texts) => ({
  vectors: texts.map(fakeVec),
  dim,
  latencyMs: 0,
}), { slot: 'resume', source: '面试问答速查.md', text: corpus, embedding: settings, apiKey: 'offline' });

console.log(`\n=== 入库：${result.chunks} 片段，维度 ${result.dim}，档位分布 ${JSON.stringify(result.byLevel)} ===`);

console.log('\n=== 检索（假向量，只验证管线与排序连通）===');
for (const q of ['你的 RAG 召回率是多少', '为什么用 RRF', '做过 Docker 部署吗', '今天天气怎么样']) {
  const r = search(store, q, fakeVec(q), { topK: 3, minScore: 0, debug: true });
  console.log(`\nQ: ${q}`);
  console.log(`   最高余弦 ${r.topScore.toFixed(4)}  低置信=${r.lowConfidence}`);
  for (const e of r.evidence.slice(0, 2)) {
    console.log(`   [${e.ref}] ${FACT_LEVEL_LABEL[e.factLevel]} cos=${e.score} kw=${e.keywordBonus} ${JSON.stringify(e.text.slice(0, 46))}`);
  }
}

rmSync(dir, { recursive: true, force: true });
console.log('\n完成。');
