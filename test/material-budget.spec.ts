/**
 * 多文档直塞预算的单元测试。
 *
 * 为什么值得单独测：三槽多文档之后，简历 + 若干项目讲稿合并起来很容易超过
 * 8000 字符的直塞预算。若退化成"先来先占"，排在前面的简历会把项目讲稿整个
 * 挤掉——而项目细节恰恰是面试官最爱追问的。这里的用例锁住"每份都留得住"。
 */
import { describe, expect, it } from 'vitest';
import { budgetDocs, parseDocBlocks, buildStablePrefix } from '../electron/llm/prompts';
import { joinSlotText } from '../shared/sessionMaterial';
import type { KbDoc } from '../shared/protocol';

const doc = (name: string, text: string): KbDoc => ({ id: name, name, text, addedAt: 0 });

describe('parseDocBlocks', () => {
  it('空串 → 空数组', () => {
    expect(parseDocBlocks('')).toEqual([]);
    expect(parseDocBlocks('   ')).toEqual([]);
  });

  it('没有标记时整体作为一份', () => {
    const r = parseDocBlocks('就是一段普通文本');
    expect(r).toHaveLength(1);
    expect(r[0].text).toBe('就是一段普通文本');
  });

  it('按【文件名】拆成多份，正文归属正确', () => {
    const merged = joinSlotText([doc('A项目.md', '甲的内容'), doc('B项目.md', '乙的内容')]);
    const r = parseDocBlocks(merged);
    expect(r.map((p) => p.name)).toEqual(['A项目.md', 'B项目.md']);
    expect(r[0].text).toBe('甲的内容');
    expect(r[1].text).toBe('乙的内容');
  });

  it('正文里出现【】但不在行首时不会误判为文件标记', () => {
    const merged = '【A.md】\n正文里提到【重点】这个词';
    const r = parseDocBlocks(merged);
    expect(r).toHaveLength(1);
    expect(r[0].name).toBe('A.md');
    expect(r[0].text).toContain('【重点】');
  });
});

describe('budgetDocs', () => {
  const PRIORITY = /(项目|经历|经验)/;

  it('无文档 → 空串', () => {
    expect(budgetDocs([], 1000, PRIORITY)).toBe('');
    expect(budgetDocs([{ name: 'a', text: '   ' }], 1000, PRIORITY)).toBe('');
  });

  it('单份且装得下 → 原文原样（不加文件名，省 token）', () => {
    expect(budgetDocs([{ name: 'a.md', text: '正文' }], 1000, PRIORITY)).toBe('正文');
  });

  it('多份且装得下 → 保留文件名标注', () => {
    const out = budgetDocs(
      [
        { name: 'A.md', text: '甲' },
        { name: 'B.md', text: '乙' },
      ],
      1000,
      PRIORITY,
    );
    expect(out).toContain('【A.md】');
    expect(out).toContain('【B.md】');
  });

  it('★超预算时每份都留得住★（不会出现"后面的文档被整个挤掉"）', () => {
    const big = '项目经历：' + '甲'.repeat(9000);
    const small = '项目经历：' + '乙'.repeat(3000);
    const out = budgetDocs(
      [
        { name: '大文档.md', text: big },
        { name: '小文档.md', text: small },
      ],
      8000,
      PRIORITY,
    );
    // 两份的文件名都必须出现——这是"按份分配"与"先来先占"的关键差别
    expect(out).toContain('【大文档.md】');
    expect(out).toContain('【小文档.md】');
    expect(out).toContain('乙');
  });

  it('每份至少拿到保底额度（600 字符量级），小文档不被饿死', () => {
    const huge = '项目经历：' + '甲'.repeat(20000);
    const tiny = '项目经历：' + '乙'.repeat(500);
    const out = budgetDocs(
      [
        { name: 'H.md', text: huge },
        { name: 'T.md', text: tiny },
      ],
      8000,
      PRIORITY,
    );
    // tiny 只有 505 字，应完整保留
    expect(out).toContain('乙'.repeat(500));
  });

  it('裁剪时优先保留命中优先级正则的段落', () => {
    const body = ['无关段落。'.repeat(80), '项目经历：我做了检索优化，召回率提升明显。', '无关段落。'.repeat(80)].join(
      '\n\n',
    );
    const out = budgetDocs([{ name: 'x.md', text: body }], 400, PRIORITY);
    expect(out).toContain('检索优化');
  });
});

describe('buildStablePrefix —— 多文档', () => {
  it('无资料时不含资料块', () => {
    const p = buildStablePrefix('', '', 'chinese');
    expect(p).not.toContain('【我的资料】');
    expect(p).not.toContain('【岗位JD】');
    expect(p).toContain('用【中文】输出');
  });

  it('多份项目资料合并后，文件名标注保留在提示词里', () => {
    const merged = joinSlotText([doc('讲稿A.md', '项目经历：甲'), doc('讲稿B.md', '项目经历：乙')]);
    const p = buildStablePrefix(merged, '', 'chinese');
    expect(p).toContain('【讲稿A.md】');
    expect(p).toContain('【讲稿B.md】');
  });

  it('标签措辞已从"简历"改为"我的资料"（该槽现在同时承载简历与项目）', () => {
    const p = buildStablePrefix('项目经历：甲', '', 'chinese');
    expect(p).toContain('【我的资料】');
    expect(p).toContain('【我的资料结束】');
  });

  it('同一输入必须产出逐字节相同的提示词（前缀缓存的前提）', () => {
    const a = buildStablePrefix('项目经历：甲', '职责：做检索', 'chinese');
    const b = buildStablePrefix('项目经历：甲', '职责：做检索', 'chinese');
    expect(a).toBe(b);
  });

  it('JD 与资料同时存在时两块都在，且 JD 在后', () => {
    const p = buildStablePrefix('项目经历：甲', '职责：做检索', 'chinese');
    expect(p.indexOf('【我的资料】')).toBeLessThan(p.indexOf('【岗位JD】'));
  });
});
