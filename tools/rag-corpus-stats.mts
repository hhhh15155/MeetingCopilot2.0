/**
 * 真实语料体检：对用户自己的长文档跑一遍切块 + 事实档位，输出统计与质量告警。
 *
 * 用法：npx vite-node tools/rag-corpus-stats.mts <语料路径> [更多路径...]
 *
 * 关注三件事：
 *   1. 切块是否会产出"只有答没有问"的孤儿段（会让 AI 无从作答）
 *   2. 事实档位分布（claimed 比例过高说明标注过激，会让模型什么都不敢讲）
 *   3. 单段长度分布（过长会撑爆注入预算，过碎会丢上下文）
 *
 * ⚠️ 档位只按**正文**判定，与 `electron/rag/indexer.ts` 保持一致：
 * 标题是导航信息，判进去会把顶层标题的风险词扩散到每一个片段。
 */
import { readFileSync, existsSync } from 'fs';
import { chunkMarkdown } from '../electron/rag/chunker';
import { classifyChunk, FACT_LEVEL_LABEL } from '../electron/rag/factlevel';
import type { FactLevel } from '../shared/protocol';

const QA = /^\s*(?:[-*>]\s*)?(?:\*\*)?(?:问|Q|问题)(?:\*\*)?\s*[：:、.．]/i;
const ANSWER = /^\s*(?:[-*>]\s*)?(?:\*\*)?(?:答|A)(?:\*\*)?\s*[：:、.．]/i;

const paths = process.argv.slice(2).filter((p) => existsSync(p));
if (paths.length === 0) {
  console.error('用法：npx vite-node tools/rag-corpus-stats.mts <语料路径> [...]');
  process.exit(2);
}

const grand = { chunks: 0, chars: 0, orphans: 0, dupQ: 0 };
const levelTotals: Record<FactLevel, number> = { verified: 0, claimed: 0, unverified: 0 };

for (const p of paths) {
  const text = readFileSync(p, 'utf8');
  const drafts = chunkMarkdown(text);
  const levels = drafts.map((d) => classifyChunk(d.text));
  const lens = drafts.map((d) => d.text.length);
  const avg = lens.reduce((a, b) => a + b, 0) / (lens.length || 1);
  const max = Math.max(...lens, 0);

  // 孤儿段：含"答："却没有任何问题标签
  const orphans = drafts.filter((d) => ANSWER.test(d.text.trim()) && !QA.test(d.text));
  // 重复问题标签：同一段里同一问题行出现两次
  const dupQ = drafts.filter((d) => {
    const lines = d.text.split('\n').map((l) => l.trim());
    const qs = lines.filter((l) => QA.test(l));
    return new Set(qs).size !== qs.length;
  });

  const byLevel: Record<FactLevel, number> = { verified: 0, claimed: 0, unverified: 0 };
  levels.forEach((l) => byLevel[l]++);

  console.log(`\n=== ${p} ===`);
  console.log(`  ${text.length} 字 → ${drafts.length} 段（平均 ${avg.toFixed(0)} 字，最长 ${max} 字）`);
  console.log(
    `  档位：已证实 ${byLevel.verified} / 目标值 ${byLevel.claimed} / 未证实 ${byLevel.unverified}` +
      `  （待确认占比 ${((byLevel.claimed / (drafts.length || 1)) * 100).toFixed(1)}%）`,
  );
  console.log(`  孤儿段（有答无问）：${orphans.length}${orphans.length ? ' ← 需关注' : ' OK'}`);
  if (orphans.length) orphans.slice(0, 3).forEach((o) => console.log(`     ${JSON.stringify(o.text.slice(0, 60))}`));
  console.log(`  重复问题标签：${dupQ.length}${dupQ.length ? ' ← 需关注' : ' OK'}`);
  if (dupQ.length) dupQ.slice(0, 3).forEach((d) => console.log(`     ${JSON.stringify(d.text.slice(0, 60))}`));

  grand.chunks += drafts.length;
  grand.chars += text.length;
  grand.orphans += orphans.length;
  grand.dupQ += dupQ.length;
  (Object.keys(byLevel) as FactLevel[]).forEach((k) => (levelTotals[k] += byLevel[k]));
}

console.log('\n=== 汇总 ===');
console.log(`  ${grand.chars} 字 → ${grand.chunks} 段`);
console.log(
  `  档位合计：已证实 ${levelTotals.verified} / 目标值 ${levelTotals.claimed} / 未证实 ${levelTotals.unverified}` +
    `（待确认占比 ${((levelTotals.claimed / (grand.chunks || 1)) * 100).toFixed(1)}%，` +
    `建议落在 10%–50%：过低=漏标，过高=用户对警示色脱敏）`,
);
console.log(`  孤儿段合计：${grand.orphans}${grand.orphans === 0 ? ' OK' : ' ← 需修复'}`);
console.log(`  重复标签合计：${grand.dupQ}${grand.dupQ === 0 ? ' OK' : ' ← 需修复'}`);
console.log(
  `\n  注入成本估算：每次提问取 top-5 段 ≈ ${Math.round((grand.chars / grand.chunks) * 5)} 字` +
    `（全量直塞需 ${grand.chars} 字，缩小约 ${Math.round(grand.chars / ((grand.chars / grand.chunks) * 5))} 倍）`,
);
