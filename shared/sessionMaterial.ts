/**
 * 会话资料的读写与迁移（三槽多文档）。
 *
 * 背景：最初是"单份知识库"，后来变成"简历 + JD 两份"，现在是
 * **简历 / 项目 / JD 三个槽位、每个槽位可放多份文档**。
 * 真实面试要同时带简历、多个项目讲稿和岗位 JD，一份一份替换太麻烦，
 * 而且项目讲稿混在简历槽里会互相稀释检索结果。
 *
 * 这个模块是**唯一**允许解释 `StoredSession` 资料字段的地方：
 *   * 旧格式（`kbText` 单槽 / `resumeText`+`jdText` 双槽）在这里一次性迁移；
 *   * 界面与主进程都只通过 `slotDocs()` 读取，不直接碰 legacy 字段。
 * 这样"兼容老文件"这件事只在一个地方做，不会散落到各处。
 */
import type { KbDoc, KbSlot, StoredSession } from './protocol';
import { KB_SLOTS } from './protocol';

/** 槽位的中文名（主进程与渲染进程都要用，放这里避免两处不一致） */
export const SLOT_LABEL_ZH: Record<KbSlot, string> = {
  resume: '简历',
  project: '项目',
  jd: '岗位 JD',
};

let docSeq = 0;
/** 生成稳定的文档 id（删除单份、逐份建索引用） */
export function newDocId(): string {
  docSeq += 1;
  return `d-${Date.now().toString(36)}-${docSeq}`;
}

/** 把一份解析好的文本包成 KbDoc */
export function makeDoc(name: string, text: string, size?: number): KbDoc {
  return { id: newDocId(), name, text, size, addedAt: Date.now() };
}

/**
 * 读取某槽位的文档列表（**已包含旧格式迁移**）。
 *
 * 迁移规则（只在 `docs` 完全为空时执行，绝不覆盖新格式）：
 *   * `kbText`（最老的单槽）→ resume 槽
 *   * `resumeText` → resume 槽；`jdText` → jd 槽
 */
export function slotDocs(session: StoredSession | null | undefined, slot: KbSlot): KbDoc[] {
  if (!session) return [];
  const existing = session.docs?.[slot];
  if (existing && existing.length > 0) return existing;
  // 新格式里这个槽是空的：看看有没有旧字段可以搬过来
  const legacy: KbDoc[] = [];
  if (slot === 'resume') {
    const text = session.resumeText ?? session.kbText;
    const name = session.resumeName ?? session.kbName;
    if (text) legacy.push({ id: `legacy-resume`, name: name || SLOT_LABEL_ZH.resume, text, addedAt: 0 });
  }
  if (slot === 'jd' && session.jdText) {
    legacy.push({ id: `legacy-jd`, name: session.jdName || SLOT_LABEL_ZH.jd, text: session.jdText, addedAt: 0 });
  }
  return legacy;
}

/** 是否还有旧格式字段需要清理（迁移后应清掉，避免与新格式并存产生歧义） */
export function hasLegacyFields(session: StoredSession): boolean {
  return !!(session.kbText || session.resumeText || session.jdText);
}

/**
 * 把老会话文件升级成三槽多文档格式，并**清掉旧字段**。
 *
 * 之所以要清掉：如果旧字段留着，下次 `slotDocs()` 会因为"新格式为空"再次
 * 迁移，删除文档后旧内容会"复活"——这是一类很难查的幽灵数据。
 */
export function migrateSessionMaterial(
  session: StoredSession,
  fallbackName: string,
): StoredSession {
  if (!hasLegacyFields(session)) return session;
  const docs: Partial<Record<KbSlot, KbDoc[]>> = { ...(session.docs ?? {}) };
  const put = (slot: KbSlot, name: string, text: string) => {
    if (!text) return;
    const cur = docs[slot] ?? [];
    // 同名的旧文档不重复添加
    if (cur.some((d) => d.name === name)) return;
    docs[slot] = [...cur, { id: `legacy-${slot}-${cur.length}`, name: name || fallbackName, text, addedAt: 0 }];
  };
  put('resume', session.resumeName ?? session.kbName ?? fallbackName, session.resumeText ?? session.kbText ?? '');
  put('jd', session.jdName ?? fallbackName, session.jdText ?? '');

  const { kbName, kbText, resumeName, resumeText, jdName, jdText, ...rest } = session;
  void kbName;
  void kbText;
  void resumeName;
  void resumeText;
  void jdName;
  void jdText;
  return { ...rest, docs };
}

/** 会话里是否一份资料都没有（用于首帧提示） */
export function hasAnyMaterial(session: StoredSession | null | undefined): boolean {
  if (!session) return false;
  return KB_SLOTS.some((s) => slotDocs(session, s).length > 0);
}

/** 汇总每槽的字符数，供界面显示"这个槽一共多少字" */
export function slotChars(session: StoredSession | null | undefined, slot: KbSlot): number {
  return slotDocs(session, slot).reduce((n, d) => n + d.text.length, 0);
}

/**
 * 把某槽位的全部文档拼成一段文本（喂给提示词/建索引用）。
 *
 * 每份前面加一行 `【文件名】`：多文档拼在一起时，模型必须知道哪段出自哪份，
 * 否则回答里会张冠李戴（把 A 项目的数据说成 B 项目的）。
 */
export function joinSlotText(docs: KbDoc[]): string {
  if (docs.length === 0) return '';
  if (docs.length === 1) return docs[0].text;
  return docs.map((d) => `【${d.name}】\n${d.text}`).join('\n\n');
}
