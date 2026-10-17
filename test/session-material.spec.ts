/**
 * 会话资料（三槽多文档）的单元测试。
 *
 * 重点在**迁移**：老会话文件里存的是单份 `kbText` 或双份 `resumeText`/`jdText`，
 * 必须无损升级到三槽多文档格式，否则老用户的资料会"凭空消失"。
 * 同时锁住一个容易踩的坑：迁移后必须清掉旧字段，否则删除文档后旧内容会"复活"。
 */
import { describe, expect, it } from 'vitest';
import {
  hasAnyMaterial,
  hasLegacyFields,
  joinSlotText,
  makeDoc,
  migrateSessionMaterial,
  newDocId,
  slotChars,
  slotDocs,
} from '../shared/sessionMaterial';
import type { KbDoc, StoredSession } from '../shared/protocol';

function session(over: Partial<StoredSession> = {}): StoredSession {
  return { id: 's1', name: '会话 1', createdAt: 0, turns: [], ...over };
}

describe('newDocId / makeDoc', () => {
  it('id 稳定且互不相同', () => {
    const a = newDocId();
    const b = newDocId();
    expect(a).not.toBe(b);
    expect(a).toMatch(/^d-/);
  });
  it('makeDoc 带上导入时间', () => {
    const d = makeDoc('讲稿.md', '正文');
    expect(d.name).toBe('讲稿.md');
    expect(d.text).toBe('正文');
    expect(d.addedAt).toBeGreaterThan(0);
  });
});

describe('slotDocs —— 新格式', () => {
  it('直接返回该槽的文档数组', () => {
    const doc = makeDoc('a.md', 'A');
    const s = session({ docs: { resume: [doc] } });
    expect(slotDocs(s, 'resume')).toEqual([doc]);
  });
  it('未设置的槽返回空数组（不返回 undefined）', () => {
    expect(slotDocs(session({ docs: { resume: [makeDoc('a.md', 'A')] } }), 'project')).toEqual([]);
    expect(slotDocs(null, 'jd')).toEqual([]);
  });
  it('每槽相互独立（三槽同时存在）', () => {
    const s = session({
      docs: {
        resume: [makeDoc('简历.md', 'R')],
        project: [makeDoc('p1.md', 'P1'), makeDoc('p2.md', 'P2')],
        jd: [makeDoc('jd.md', 'J')],
      },
    });
    expect(slotDocs(s, 'resume')).toHaveLength(1);
    expect(slotDocs(s, 'project')).toHaveLength(2);
    expect(slotDocs(s, 'jd')).toHaveLength(1);
  });
});

describe('slotDocs —— 旧格式回退（读得到，不改数据）', () => {
  it('最老的单槽 kbText → resume 槽', () => {
    const s = session({ kbName: '资料.md', kbText: '旧正文' });
    const docs = slotDocs(s, 'resume');
    expect(docs).toHaveLength(1);
    expect(docs[0].text).toBe('旧正文');
    expect(docs[0].name).toBe('资料.md');
  });
  it('双槽 resumeText / jdText 各归其位', () => {
    const s = session({ resumeName: '简历.md', resumeText: 'R', jdName: 'jd.md', jdText: 'J' });
    expect(slotDocs(s, 'resume')[0].text).toBe('R');
    expect(slotDocs(s, 'jd')[0].text).toBe('J');
    expect(slotDocs(s, 'project')).toEqual([]);
  });
  it('新格式为空时才回退（新格式优先，不会被旧字段覆盖）', () => {
    const s = session({
      docs: { resume: [makeDoc('新.md', '新内容')] },
      resumeText: '旧内容',
    });
    expect(slotDocs(s, 'resume')[0].text).toBe('新内容');
  });
});

describe('migrateSessionMaterial', () => {
  it('单槽 kbText 迁到 resume 并清掉旧字段', () => {
    const s = migrateSessionMaterial(session({ kbName: '资料.md', kbText: '旧正文' }), '资料');
    expect(slotDocs(s, 'resume')[0].text).toBe('旧正文');
    expect(hasLegacyFields(s)).toBe(false);
    expect(s.kbText).toBeUndefined();
    expect(s.kbName).toBeUndefined();
  });

  it('双槽迁到 resume / jd 并清掉旧字段', () => {
    const s = migrateSessionMaterial(
      session({ resumeName: '简历.md', resumeText: 'R', jdName: 'jd.md', jdText: 'J' }),
      '资料',
    );
    expect(slotDocs(s, 'resume')[0].text).toBe('R');
    expect(slotDocs(s, 'jd')[0].text).toBe('J');
    expect(hasLegacyFields(s)).toBe(false);
  });

  it('保留新格式已有的文档（迁移是合并，不是覆盖）', () => {
    const s = migrateSessionMaterial(
      session({ docs: { project: [makeDoc('p.md', 'P')] }, resumeText: 'R' }),
      '资料',
    );
    expect(slotDocs(s, 'project')).toHaveLength(1);
    expect(slotDocs(s, 'resume')[0].text).toBe('R');
  });

  it('★关键★ 迁移后删除文档，旧内容不会"复活"', () => {
    // 若迁移只搬数据不清旧字段，slotDocs() 会因为"新格式为空"再次回退到旧字段，
    // 于是用户删掉文档后旧内容又出现了——这是最难查的一类幽灵数据。
    const migrated = migrateSessionMaterial(session({ resumeText: 'R' }), '资料');
    const deleted: StoredSession = { ...migrated, docs: {} };
    expect(slotDocs(deleted, 'resume')).toEqual([]);
  });

  it('无旧字段时原样返回（不产生无谓的对象拷贝）', () => {
    const s = session({ docs: { resume: [makeDoc('a.md', 'A')] } });
    expect(migrateSessionMaterial(s, '资料')).toBe(s);
  });

  it('空文本的旧字段不产生空文档', () => {
    const s = migrateSessionMaterial(session({ resumeText: '', jdText: '' }), '资料');
    expect(hasAnyMaterial(s)).toBe(false);
  });

  it('旧字段没有文件名时用回退名', () => {
    const s = migrateSessionMaterial(session({ resumeText: 'R' }), '我的资料');
    expect(slotDocs(s, 'resume')[0].name).toBe('我的资料');
  });
});

describe('hasAnyMaterial / slotChars', () => {
  it('三个槽都空 → false；任一有 → true', () => {
    expect(hasAnyMaterial(session())).toBe(false);
    expect(hasAnyMaterial(session({ docs: { jd: [makeDoc('j.md', 'J')] } }))).toBe(true);
    expect(hasAnyMaterial(session({ resumeText: 'R' }))).toBe(true);
  });
  it('slotChars 累加该槽全部文档', () => {
    const s = session({ docs: { project: [makeDoc('a', '12345'), makeDoc('b', '678')] } });
    expect(slotChars(s, 'project')).toBe(8);
    expect(slotChars(s, 'jd')).toBe(0);
  });
});

describe('joinSlotText', () => {
  const d = (name: string, text: string): KbDoc => ({ id: name, name, text, addedAt: 0 });

  it('空数组 → 空串', () => {
    expect(joinSlotText([])).toBe('');
  });
  it('单份直接返回原文（不加文件名前缀，省 token）', () => {
    expect(joinSlotText([d('a.md', '正文')])).toBe('正文');
  });
  it('★多份必须带文件名★ 否则模型会把 A 项目的数据说成 B 项目的', () => {
    const out = joinSlotText([d('A项目.md', 'A 的内容'), d('B项目.md', 'B 的内容')]);
    expect(out).toContain('【A项目.md】');
    expect(out).toContain('【B项目.md】');
    expect(out.indexOf('A 的内容')).toBeLessThan(out.indexOf('B 的内容'));
  });
});
