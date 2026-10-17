/**
 * 切块器与事实边界标注的单元测试。
 * 这两块是"防瞎编"的地基：切错了会丢标注，标错了会放行包装数字。
 */
import { describe, expect, it } from 'vitest';
import { chunkMarkdown, chunkLabel, splitSections } from '../electron/rag/chunker';
import { classifyChunk, classifySentence, factLevelBreakdown } from '../electron/rag/factlevel';

describe('chunker / splitSections', () => {
  it('保留多级标题路径', () => {
    const md = ['# 第一部分 自我介绍', '正文A', '## （1）一分钟版本', '正文B'].join('\n');
    const secs = splitSections(md);
    expect(secs.map((s) => s.heading)).toEqual([
      '第一部分 自我介绍',
      '第一部分 自我介绍 / （1）一分钟版本',
    ]);
    expect(secs[0].body).toContain('正文A');
  });

  it('标题词保留在正文里（检索强信号）', () => {
    const secs = splitSections('# RAG 实现细节\n内容');
    expect(secs[0].body).toContain('RAG 实现细节');
  });
});

describe('chunker / chunkMarkdown', () => {
  it('短文档产出单个片段', () => {
    const out = chunkMarkdown('# 标题\n\n一段不长的正文。');
    expect(out).toHaveLength(1);
    expect(out[0].heading).toBe('标题');
  });

  it('超长正文按中文句末标点切分', () => {
    // 20 个句子，每句约 20 字 → 总长 400+，切成多块
    const body = Array.from({ length: 20 }, (_, i) => `这是第${i}句用来测试切分的中文内容。`).join('');
    const out = chunkMarkdown(`# 长文\n\n${body}`, { chunkChars: 150, overlapChars: 0 });
    expect(out.length).toBeGreaterThan(1);
    for (const c of out) expect(c.text.length).toBeLessThanOrEqual(200);
  });

  it('不把一组问答切散（问：/答：成对保留）', () => {
    const qa = [
      '问：你这个项目里 RAG 的召回率是多少？',
      '答：我实测是 88.7%，用的是 500 条测试集。',
      '问：那你是怎么评估的？',
      '答：我用的是人工标注的 11 条样本做 baseline，准确率 81.8%。',
    ].join('\n');
    const out = chunkMarkdown(`## 面试问答\n\n${qa}`, { chunkChars: 60, overlapChars: 0 });
    // 每个片段都不应以"答："开头却缺少对应"问："
    for (const c of out) {
      if (c.text.includes('答：')) expect(c.text).toContain('问：');
    }
    // 内容不丢失
    const joined = out.map((c) => c.text).join('');
    expect(joined).toContain('88.7%');
    expect(joined).toContain('81.8%');
  });

  it('序号连续且从 0 开始', () => {
    const body = Array.from({ length: 30 }, () => '这是一句比较长的中文测试内容用于切块。').join('');
    const out = chunkMarkdown(`# T\n\n${body}`, { chunkChars: 120, overlapChars: 20 });
    out.forEach((c, i) => expect(c.index).toBe(i));
  });

  it('chunkLabel 生成可读标签', () => {
    expect(chunkLabel('讲稿.md', '第3部分', 0)).toBe('讲稿.md · 第3部分 · 第1段');
  });
});

describe('factlevel', () => {
  it('识别简历里的包装数字为 claimed', () => {
    expect(classifySentence('简历里写了 Accuracy 91.3%，用的是约 500 条测试集。')).toBe('claimed');
  });

  it('识别自承认的边界声明为 claimed', () => {
    expect(classifySentence('简历里的 500 条 / Accuracy 91.3% 在仓库中没有支撑。')).toBe('claimed');
    expect(classifySentence('【边界】这部分我没有实际做过。')).toBe('claimed');
  });

  it('识别规划口吻为 unverified', () => {
    expect(classifySentence('后续计划接入 vLLM 做量化推理优化。')).toBe('unverified');
    expect(classifySentence('预计吞吐能提升 3 倍。')).toBe('unverified');
  });

  it('识别可指向代码的事实为 verified', () => {
    expect(classifySentence('在 electron/rag/search.ts 里实现了 RRF 融合。')).toBe('verified');
    expect(classifySentence('【项目】我负责了检索模块的实现。')).toBe('verified');
  });

  it('整块取最高风险档位（保守优先）', () => {
    const mixed = [
      '我在 oceanagent 项目里实现了多 Agent 协作。',
      '简历上写的是准确率 91.3%，测试集 500 条。',
    ].join('\n');
    expect(classifyChunk(mixed)).toBe('claimed');
  });

  it('无风险内容保持 verified', () => {
    expect(classifyChunk('我负责了数据清洗和特征工程这两块工作。')).toBe('verified');
  });

  it('breakdown 统计各档位句子数', () => {
    const b = factLevelBreakdown('后续计划做压测。\n我在 src/a.ts 里实现了接口。');
    expect(b.unverified).toBeGreaterThanOrEqual(1);
    expect(b.verified).toBeGreaterThanOrEqual(1);
  });
});
