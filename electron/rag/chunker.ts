/**
 * 中文优先的 Markdown 切块器（MeetingCopilot 2.0）。
 *
 * 借鉴 NexQ `rag/chunker.rs` 的递归切分思路，但针对中文面试资料重写：
 *   * NexQ 用 `\n\n / \n / ". " / "? " / "! " / " "` 作分隔符——中文里
 *     ". " / "? " / "! " 几乎不出现（中文没有词间空格），实际退化成按换行切。
 *   * 这里改为中文标点 `。！？；` + 段落空行，并把 **分隔符保留在片段内**，
 *     否则 STAR 结构里的"结果："常被切掉。
 *   * 长度以**字符**计而非 token：中文 1 字≈1 token，`chars/4` 那种估算对
 *     中文严重低估（会切出远超模型窗口的块）。
 *
 * 结构化保留（防瞎编的关键）：
 *   * 维护 `## 标题` 路径，作为片段的 heading 前缀——检索命中时能说清
 *     "这段出自第 3 部分"，也显著提升召回（标题词是强信号）。
 *   * 绝不把一组"问：…/答：…"切散：检测到问答对时整对作为一个单元。
 */

/** 目标片段长度（字符）。中文 1 字≈1 token，700 足够容纳一个完整问答。 */
export const DEFAULT_CHUNK_CHARS = 700;
/** 相邻片段重叠（字符），避免答案被切断在边界 */
export const DEFAULT_OVERLAP_CHARS = 120;
/** 小于此长度的片段并入前一个，避免产生无信息碎片 */
const MIN_CHUNK_CHARS = 60;

export interface ChunkDraft {
  /** 片段正文（含 heading 前缀之后的纯文本） */
  text: string;
  /** 所属标题路径，如 "第 3 部分 项目细节 / （2）RAG 实现" */
  heading: string;
  /** 在原文中的序号（0 基，切分后） */
  index: number;
}

export interface ChunkOptions {
  chunkChars?: number;
  overlapChars?: number;
}

interface Section {
  heading: string;
  body: string;
}

const HEADING_RE = /^(#{1,6})\s+(.*)$/;
/** 代码围栏：``` 或 ~~~（可带语言标注） */
const FENCE_RE = /^\s*(```|~~~)/;
/**
 * 问答单元起始标志。必须同时支持：
 *   问：… / Q: … / 问题：…
 *   **问**：…（后缀式粗体）  **问：**（前缀式粗体）
 *   - 问：… / > 问：…（列表或引用）
 * 对抗性验证的对照实验：plain 破损 0 处、bold 破损 1 处——
 * 只覆盖前缀式粗体不够，真实讲稿两种写法都会出现。
 */
const QA_START_RE = /^\s*(?:[-*>]\s*)?(?:\*\*)?(?:问|Q|问题)(?:\*\*)?\s*[：:、.．]/i;

/** 把 Markdown 按标题切成 section，每段带完整标题路径。 */
export function splitSections(md: string): Section[] {
  const lines = md.split(/\r?\n/);
  const stack: string[] = [];
  const out: Section[] = [];
  let buf: string[] = [];
  let current = '';
  /** 是否在代码围栏内。围栏里的 `# 注释` 不是标题—— */
  let inFence = false;

  const flush = () => {
    const body = buf.join('\n').trim();
    if (body) out.push({ heading: current, body });
    buf = [];
  };

  for (const line of lines) {
    if (FENCE_RE.test(line)) {
      // 围栏行本身保留在正文里（代码是资料内容的一部分）
      inFence = !inFence;
      buf.push(line);
      continue;
    }
    // Python/shell 注释 `# 说明` 与 Markdown 标题同形，
    // 不跟踪围栏会让标题路径被代码注释污染（实测会出现
    // heading = "计算准确率" 这种来自代码注释的假标题）。
    const m = inFence ? null : HEADING_RE.exec(line);
    if (m) {
      flush();
      const level = m[1].length;
      const title = m[2].trim();
      stack.length = level - 1;
      stack[level - 1] = title;
      current = stack.filter((s) => s !== undefined && s !== '').join(' / ');
      // 标题本身也留在正文里：检索时标题词是强信号
      buf.push(title);
      continue;
    }
    buf.push(line);
  }
  flush();
  return out;
}

/**
 * 把一个 section 的正文切成若干片段。
 * 先按段落，再按中文句末标点，最后硬切——三级递退，保证一定能切出来。
 *
 * @param qLabel 若该正文属于某个问答组，传入"问：…"原文。
 *   每个不含问题标签的片段都会自动补上 `【续】问：…` 前缀。
 *
 *   **标签必须在"单元"层面就位，而不是在最后补**：切分是多层的
 *   （段落层 → 句子层 → 硬切），事后补标签会被下一层再次切散，
 *   产生"只有答没有问"的孤儿片段——实测反复复现过。
 */
function splitBody(body: string, chunkChars: number, qLabel?: string): string[] {
  const text = body.trim();
  /** 相对容量：标签本身已占掉一段额度，必须扣掉，否则小 chunkChars 下必溢。 */
  const cap = qLabel ? Math.max(40, chunkChars - qLabel.length - 4) : chunkChars;

  /** 给片段打标签（已带问题标签的原样返回）。 */
  const label = (p: string): string =>
    !qLabel || QA_START_RE.test(p.trim()) ? p : `${qLabel}\n${p}`;

  // 单元构建：每个单元在产出时就确保不超过 cap 且已带标签
  const units: string[] = [];
  const emitSlices = (t: string) => {
    for (let i = 0; i < t.length; i += cap) units.push(label(t.slice(i, i + cap)));
  };

  for (const p of text.split(/\n{2,}/)) {
    const t = p.trim();
    if (!t) continue;
    if (t.length <= cap) {
      units.push(label(t));
      continue;
    }
    // 段落过长 → 按中文句末标点切（保留标点）
    const sentences = t.split(/(?<=[。！？；!?;])/).map((s) => s.trim()).filter(Boolean);
    if (sentences.length <= 1) {
      emitSlices(t); // 一句超长（整段无标点）→ 硬切
      continue;
    }
    for (const s of sentences) {
      if (s.length <= cap) units.push(label(s));
      else emitSlices(s);
    }
  }
  if (units.length === 0) return [label(text)];

  // 合并单元到接近 cap 的片段；本身已达/超 cap 的单元不再拼接
  const chunks: string[] = [];
  let cur = '';
  for (const u of units) {
    if (u.length >= cap) {
      if (cur) {
        chunks.push(cur);
        cur = '';
      }
      chunks.push(u);
      continue;
    }
    if (cur && cur.length + u.length + 1 > cap) {
      chunks.push(cur);
      cur = u;
    } else {
      cur = cur ? `${cur}\n${u}` : u;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

/**
 * 主入口：把一份 Markdown 切成带标题路径与序号的片段。
 *
 * 问答保护：如果 section 里出现"问：/Q：/**问**："起始的问答对，按问答对
 * 边界聚合，不跨对切分（一组问答是一个完整语义单元，切开后 AI 只会看到半句）。
 *
 * 更关键的是**长答案的续段**：一组问答超过 chunkChars 时，后半段会脱离问题
 * 标签，出现"只有答没有问"的孤儿片段——AI 看到一段没有问题的答案，根本
 * 不知道该回答什么。所以拆分后给续段补上 `【续】问：…` 前缀。
 */
export function chunkMarkdown(md: string, opts: ChunkOptions = {}): ChunkDraft[] {
  const chunkChars = opts.chunkChars ?? DEFAULT_CHUNK_CHARS;
  const overlap = opts.overlapChars ?? DEFAULT_OVERLAP_CHARS;
  const sections = splitSections(md);
  const out: ChunkDraft[] = [];

  for (const sec of sections) {
    const body = sec.body;
    const bodyLines = body.split('\n');

    // 找出每个问答组的起点
    const starts: number[] = [];
    bodyLines.forEach((l, i) => {
      if (QA_START_RE.test(l)) starts.push(i);
    });

    let pieces: string[];
    if (starts.length === 0) {
      pieces = splitBody(body, chunkChars);
    } else {
      // 按问答组重排，**每个组自带问题标签**，绝不产出孤儿答案段。
      //
      // 两条硬性约束（都来自实测踩坑）：
      //  1. 不能把结果整体再过一遍 splitBody —— 会二次切散标签。
      //  2. "问题行 + 答案行"必须当**原子单元**：拆开它们一定产生
      //     "只有答没有问"的孤儿段（AI 拿到一段没有问题的答案无从作答）。
      //     所以只有当单个问答组确实超长、且内部存在可安全切分的
      //     *段落*时，才拆这一组（此时 splitBody 会给每片补标签）。
      const annotated: string[] = [];
      const preamble = bodyLines.slice(0, starts[0]).join('\n').trim();
      if (preamble) annotated.push(preamble);

      for (let k = 0; k < starts.length; k++) {
        const from = starts[k];
        const to = k + 1 < starts.length ? starts[k + 1] : bodyLines.length;
        const group = bodyLines.slice(from, to).join('\n').trim();
        if (!group) continue;
        const qLabel = bodyLines[from].trim();

        // 把"问题行 + 紧邻的答案行"视为不可拆的整体：
        // 同行换行切在这里是安全的，切在两者之间则会产生孤儿段。
        const lines = group.split('\n').map((l) => l.trim()).filter(Boolean);
        const atomic = [lines[0] ?? '', lines[1] ?? ''].join('\n');

        // 只有整组超长、且"问题+首答"这个原子块本身放得下，才允许拆。
        // 否则整组带标签一起产出（宁可片段略长，也不能让答案失去问题）。
        const canSplit = group.length > chunkChars && atomic.length <= chunkChars;

        if (!canSplit) {
          // group 的首行本来就是 qLabel，**不要再拼一次**——
          // 否则提示词里会出现两行相同的问题（浪费 token 且显得重复）。
          annotated.push(group);
          continue;
        }
        for (const p of splitBody(group, chunkChars, qLabel)) annotated.push(p);
      }

      pieces = annotated;
    }

    for (const piece of pieces) {
      const t = piece.trim();
      if (!t) continue;
      // 过短的尾巴并入上一片段（同一 section 内）
      const prev = out[out.length - 1];
      if (t.length < MIN_CHUNK_CHARS && prev && prev.heading === sec.heading) {
        prev.text = `${prev.text}\n${t}`;
        continue;
      }
      out.push({ text: t, heading: sec.heading, index: out.length });
    }
  }

  // 重叠：把上一片段尾部若干字符前置到本片段（同 section 内才有意义）
  if (overlap > 0 && out.length > 1) {
    for (let i = 1; i < out.length; i++) {
      if (out[i].heading !== out[i - 1].heading) continue;
      const tail = out[i - 1].text.slice(-overlap);
      if (tail && !out[i].text.startsWith(tail)) out[i].text = `…${tail}\n${out[i].text}`;
    }
  }

  return out.map((c, i) => ({ ...c, index: i }));
}

/** 生成注入 prompt 时用的可读标签：`文件名 · 标题路径 · 第N段`。 */
export function chunkLabel(source: string, heading: string, index: number): string {
  const h = heading ? ` · ${heading}` : '';
  return `${source}${h} · 第${index + 1}段`;
}
