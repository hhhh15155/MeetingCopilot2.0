/**
 * Prompt construction for meeting/interview answering (pure logic, TDD).
 * R4: (a) manual — answer THIS sentence; (b) continuous — advise on recent
 * speech; (c) free — ask over the conversation; (d) translate — translate a
 * line to Chinese. Answer language (zh/en) is a runtime prompt hook.
 *
 * v2 (2026-07-10): cache-friendly three-layer layout —
 *   stable prefix  = persona + 【简历】 + 【岗位JD】 + lang directive
 *                    (BYTE-STABLE across requests → DeepSeek prefix cache)
 *   slow state     = 【面试备忘】memo (updated every few turns)
 *   fast context   = history turns + recent transcript + this question + hint
 *
 * v3 (MeetingCopilot 2.0): RAG 证据层 + 分层兜底 —
 *   证据(evidence)是**第 3.5 层**：每轮都随问题变，所以只能插在 memo 之后、
 *   history 之前。放到 stable prefix 里会让公共前缀每轮全失效，DeepSeek 的
 *   前缀缓存（0.1x 价格 + 更快首字）就白建了。
 *   `buildFallbackAnswer` 覆盖"根本不调用 LLM"的三条路径（没索引/没命中/检索失败），
 *   这些话术是面试现场要照念的，所以禁 Markdown、必须第一人称口语。
 */
import type { ChatMessage } from './adapter';
import type { FactLevel, RagEvidence } from '../../shared/protocol';
import { FACT_LEVEL_LABEL, FACT_LEVEL_RULE } from '../rag/factlevel';
import { classifyQuestion, isLikelyQuestion, type QuestionKind } from '../../shared/textHeuristics';

export { isLikelyQuestion, classifyQuestion };

export const MAX_CONTEXT_CHARS = 2400;

export type AnswerLang = 'chinese' | 'english';

/** The prompt hook that steers DeepSeek's reply language (R: 模式选择). */
export function langDirective(lang: AnswerLang): string {
  return lang === 'english'
    ? '- 用【英文】输出我要念的话；必要时在最后附一句极简中文备注。'
    : '- 用【中文】输出。';
}

/** teleprompter persona: the output IS what the user reads aloud, verbatim */
const PERSONA = [
  '你是我的实时面试提词器。我正在参加面试，屏幕上是面试官说话的实时转录。',
  '你输出的内容就是我接下来要照着念的话，必须遵守：',
  '- 全程用第一人称「我」，口语自然，让我可以一字不改地念出来；',
  '- 第一句先给结论或直接回应，再展开 2-3 个短要点；',
  '- 全文控制在 30-60 秒内可念完（约 150-350 字）；',
  '- 不用 Markdown 标题、编号、加粗等书面格式，分点直接换行；',
  '- 行为/经历类问题按 STAR 展开：情境→任务→行动→结果；',
  '- 技术类问题先一句话讲思路，再给关键点，必要时给复杂度或对比结论；',
  '- 只能使用【简历】里的真实经历，绝不编造简历之外的公司、项目、数字；',
  '- 没把握的问题，给出稳妥的通用说法，或一句得体的争取思考时间的话术。',
];

/** total injected background budget; keeps prompts bounded regardless of size */
export const MAX_BACKGROUND_CHARS = 8000;
/** when both slots are present the resume gets the bigger share */
export const RESUME_BUDGET = 5000;
export const JD_BUDGET = MAX_BACKGROUND_CHARS - RESUME_BUDGET;

/** resume: keep project/work-experience sections when over budget */
export const RESUME_PRIORITY =
  /(项目|经历|经验|工作|实习|成果|职责|Project|Experience|Work|Achievement)/i;
/** JD: keep responsibilities/requirements sections when over budget */
export const JD_PRIORITY =
  /(职责|要求|责任|任职|资格|技能|优先|加分|Responsibilit|Requirement|Qualification|Skill)/i;

/**
 * Deterministic budget clip that prefers paragraphs matching `priority`
 * (e.g. a resume's project experience, a JD's requirements) instead of a
 * blind head-truncation. Output order stays the original document order.
 */
export function smartClip(text: string, budget: number, priority: RegExp): string {
  const t = text.trim();
  if (t.length <= budget) return t;
  const paras = t.split(/\n{2,}/);
  const picked = new Set<number>();
  let used = 0;
  const tryTake = (i: number) => {
    if (picked.has(i)) return;
    const cost = paras[i].length + 2; // + join separator
    if (used + cost > budget) return;
    picked.add(i);
    used += cost;
  };
  for (let i = 0; i < paras.length; i++) if (priority.test(paras[i])) tryTake(i);
  for (let i = 0; i < paras.length; i++) tryTake(i);
  if (picked.size === 0) return t.slice(0, budget); // one giant paragraph
  return paras
    .map((p, i) => (picked.has(i) ? p : null))
    .filter((p): p is string => p !== null)
    .join('\n\n');
}

/**
 * 把"多份文档拼接文本"拆回逐份，供 `budgetDocs` 按份分配预算。
 *
 * 拼接格式由 `shared/sessionMaterial.joinSlotText()` 决定：多份时每份前面有一行
 * `【文件名】`。单份时没有标记，整体作为一份返回。
 *
 * 这样做是为了不改动 IPC 载荷形状（仍然是两个字符串）：主进程拿到的合并文本里
 * 已经带了足够信息，能还原出"有哪几份、各多长"，从而按比例分配直塞额度。
 */
export function parseDocBlocks(merged: string): { name: string; text: string }[] {
  const t = merged.trim();
  if (!t) return [];
  const re = /^【([^】\n]{1,120})】$/gm;
  const marks: { name: string; start: number; end: number }[] = [];
  for (const m of t.matchAll(re)) {
    marks.push({ name: m[1], start: m.index ?? 0, end: (m.index ?? 0) + m[0].length });
  }
  if (marks.length === 0) return [{ name: '资料', text: t }];
  return marks.map((mk, i) => ({
    name: mk.name,
    text: t.slice(mk.end, i + 1 < marks.length ? marks[i + 1].start : t.length).trim(),
  }));
}

/**
 * 多文档直塞的预算分配。
 *
 * 背景：现在每个槽位可以放多份文档（简历 + 若干项目讲稿 + JD），合并后很容易
 * 远超直塞预算。若按原来的做法"先来先占"，排在前面的简历会把后面的项目讲稿
 * 整个挤掉——而项目细节恰恰是面试官最爱追问的部分。
 *
 * 做法：**按比例分配**，并给每份保底 MIN 字符，避免小文档被大文档饿死。
 * 超预算的文档用 `smartClip` 按各自的优先级正则裁剪。
 *
 * @param parts 每份文档的 `{name, text}`
 * @param budget 总字符预算
 * @param priority 该槽位的优先级正则（简历看"项目/经历"，JD 看"职责/要求"）
 */
export function budgetDocs(
  parts: { name: string; text: string }[],
  budget: number,
  priority: RegExp,
): string {
  const alive = parts.filter((p) => p.text.trim());
  if (alive.length === 0) return '';
  const total = alive.reduce((n, p) => n + p.text.length, 0);
  if (total <= budget) {
    // 装得下就整份给，但多份时仍标注文件名（模型要知道哪段出自哪份）
    return alive.length === 1 ? alive[0].text : alive.map((p) => `【${p.name}】\n${p.text}`).join('\n\n');
  }
  // 每份保底 600 字符；剩余按"文档数量均分"，避免超大文档独占
  const MIN = 600;
  const share = Math.max(MIN, Math.floor(budget / alive.length));
  return alive
    .map((p) => `【${p.name}】\n${smartClip(p.text, share, priority)}`)
    .join('\n\n');
}

/**
 * The BYTE-STABLE system prompt: persona + resume + JD + language directive.
 * Same inputs MUST yield the identical string (no timestamps / randomness) —
 * the LLM prewarm request and every real request share this prefix so the
 * provider's prefix cache (DeepSeek 0.1x pricing + faster prefill) hits.
 */
export function buildStablePrefix(resume: string, jd: string, lang: AnswerLang): string {
  const parts = [...PERSONA];
  const r = resume.trim();
  const j = jd.trim();
  if (r) {
    parts.push(
      '',
      // 多份文档时标签改成"我的资料"：现在这个槽同时承载简历与项目讲稿，
      // 写死"简历"会让模型把项目细节当成简历内容，措辞上也会别扭。
      '【我的资料】（简历与项目资料，我的真实经历，回答只能基于此）',
      budgetDocs(parseDocBlocks(r), j ? RESUME_BUDGET : MAX_BACKGROUND_CHARS, RESUME_PRIORITY),
      '【我的资料结束】',
    );
  }
  if (j) {
    parts.push(
      '',
      '【岗位JD】（本场面试针对的职位，回答向它贴合）',
      budgetDocs(parseDocBlocks(j), r ? JD_BUDGET : MAX_BACKGROUND_CHARS, JD_PRIORITY),
      '【岗位JD结束】',
    );
  }
  parts.push('', langDirective(lang));
  return parts.join('\n');
}

/** one advisory line appended to the user message; '' when unknown */
export function questionHint(kind: QuestionKind): string {
  switch (kind) {
    case 'behavioral':
      return '（题型：行为/经历题——用 STAR 结构，讲简历里的真实经历）';
    case 'technical':
      return '（题型：技术题——先一句话思路，再关键点，必要时给复杂度）';
    case 'smalltalk':
      return '（题型：寒暄/暖场——一两句自然简短的回应即可，不用展开）';
    default:
      return '';
  }
}

// ---------- P1-5: rolling interview memo (consistency > compression) ----------

/** hard bound on the stored memo (prompt asks for ≤800, clamp defends) */
export const MAX_MEMO_CHARS = 1000;

export function clampMemo(text: string): string {
  const t = text.trim();
  return t.length > MAX_MEMO_CHARS ? t.slice(0, MAX_MEMO_CHARS) : t;
}

/**
 * Fold one finished Q&A into the rolling memo (async, off the critical path).
 * The memo keeps the interview self-consistent: what was asked, what I have
 * claimed as fact, what the interviewer cares about.
 */
export function buildMemoUpdateMessages(oldMemo: string, question: string, answer: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content: [
        '你是面试会话的备忘维护器。把新一轮问答合并进备忘，输出更新后的完整备忘。',
        '备忘不超过 800 字，固定四节（无内容的节保留标题写「无」）：',
        '【已问问题】每题一行，最新在最后',
        '【我已声称的事实】数字、经历、立场——后续回答绝不能与之矛盾',
        '【面试官关注点】从提问推断',
        '【注意事项】答得不稳的点、需要圆回来的坑',
        '合并去重；超长时优先丢最旧的已问问题。只输出备忘本身，不要任何解释。',
      ].join('\n'),
    },
    {
      role: 'user',
      content: `【当前备忘】\n${oldMemo.trim() || '（空）'}\n\n【新一轮问答】\n问：${question.trim()}\n答：${answer.trim()}`,
    },
  ];
}

// ---------- RAG 证据注入（第 3.5 层：memo 之后、history 之前） ----------

/**
 * 一条证据渲染成 `[S1]（来源：讲稿.md · 第3部分 · 已证实）正文…`。
 * 档位文案与处理规则都来自 factlevel.ts 的单一事实源，避免两处措辞漂移。
 */
function evidenceLine(e: RagEvidence, ref: string): string {
  const src = (e.source ?? '').trim();
  const heading = (e.heading ?? '').trim();
  const level = FACT_LEVEL_LABEL[e.factLevel] || FACT_LEVEL_LABEL.verified;
  const meta = [src ? `来源：${src}` : '', heading, level].filter(Boolean).join(' · ');
  return `[${ref}]（${meta}）${(e.text ?? '').trim()}`;
}

/**
 * 把检索命中的片段变成**一条独立的 user 消息**（+ assistant 确认）。
 *
 * 位置由调用方保证：必须插在 memo 之后、history 之前（见 buildAnswerMessages）。
 * 把这段放进 stable prefix 会让系统提示每轮都变，前缀缓存彻底失效。
 *
 * 硬性约束的措辞刻意写得**可执行**："资料里没有的不要补" 而不是
 * "不要编造" —— 后者会被模型理解成"编得像真的就行"。
 */
export function buildEvidenceMessages(
  evidence: RagEvidence[],
  opts: { requireCitations: boolean },
): ChatMessage[] {
  const rows = (evidence ?? [])
    .filter((e): e is RagEvidence => !!e && typeof e.text === 'string' && e.text.trim().length > 0)
    .map((e, i) => ({ e, ref: (e.ref ?? '').trim() || `S${i + 1}` }));
  if (rows.length === 0) return [];

  // 档位相同的片段共用一条规则（规则文本较长，逐条重复五遍会白白多花 prompt token；
  // 规则行上的编号列表保证每条片段都能对上自己的规则）
  const order: FactLevel[] = [];
  const byLevel = new Map<FactLevel, string[]>();
  for (const { e, ref } of rows) {
    const lv = FACT_LEVEL_LABEL[e.factLevel] ? e.factLevel : 'verified';
    const refs = byLevel.get(lv);
    if (refs) refs.push(`[${ref}]`);
    else {
      byLevel.set(lv, [`[${ref}]`]);
      order.push(lv);
    }
  }

  const lines: string[] = [
    '【可用资料片段】（本轮回答只能用这些片段里的事实，编号 [S1][S2] 供引用）',
    ...rows.map(({ e, ref }) => evidenceLine(e, ref)),
    '',
    // 优先级声明：上面 persona 写的是"只能使用【简历】里的真实经历"，而【简历】
    // 未必存在（很多场次只导入了讲稿/项目手册，简历槽是空的）。不点明的话，
    // 模型会因为"我的资料里没有这段"而不敢使用检索到的片段，
    // 白白浪费掉整条 RAG 链路——这是最容易发生、也最难自查的一类失效。
    '【与其它资料的关系】上面这些片段就是我的真实资料（简历/项目讲稿/手册，按文件名与标题区分）。',
    '若同时给了【简历】或【岗位JD】，它们与这些片段同等可信、都可用；凭它们作答就是基于真实经历，不算编造。',
    '',
    '【档位处理规则】',
    ...order.map((lv) => `· ${FACT_LEVEL_LABEL[lv]}（${byLevel.get(lv)?.join('、') ?? ''}）：${FACT_LEVEL_RULE[lv]}`),
    '',
    '【硬性约束】',
    '1. 每一句事实性陈述都必须能指到上面某个编号（如 [S1]）；指不到来源的事实句，一句都不要写出来。',
    `2. 标着「${FACT_LEVEL_LABEL.claimed}」的片段里的数字必须原样引用，禁止换算、取整、外推、四舍五入，也不许加「约」「左右」这类模糊词。`,
    `3. 标着「${FACT_LEVEL_LABEL.unverified}」的片段只能讲成计划或预期（「我计划…」「我打算…」），不许讲成已经完成、已经上线、已经达成。`,
    '4. 资料里没有的能力、公司、项目、数字，一律不要补；宁可说「这块我没做过」，也不要顺着问题编一个差不多的。',
    `5. 只有标着「${FACT_LEVEL_LABEL.verified}」的片段可以当既成事实讲；其余档位按第 2、3 条处理。`,
  ];
  if (opts.requireCitations) {
    lines.push(
      '6. 输出里每处事实都要带来源编号，直接写在句子后面，如 [S1]；一句话用到两处就写 [S1][S2]。没带编号的句子只能是非事实性的过渡话。',
    );
  }
  lines.push('这些限制只决定你说什么，不改变怎么说：仍然是第一人称、口语、可以直接念的话。');

  return [
    { role: 'user', content: lines.join('\n') },
    { role: 'assistant', content: '收到，我只用这些片段里的事实作答，片段里没有的不补，档位按上面的规则处理。' },
  ];
}

// ---------- 分层兜底话术（防瞎编的最后一道闸：根本不调用 LLM） ----------

/** 兜底原因：三种"不能让模型自由发挥"的情形。 */
export type FallbackReason =
  /** 知识库为空 / 未建索引 */
  | 'no_index'
  /** 检索到了，但最高相似度低于阈值（资料里没有相关内容） */
  | 'low_score'
  /** 检索失败（网络 / API 报错）或结果为空 */
  | 'no_evidence';

export interface FallbackContext {
  /** 面试官刚问的话。只用来决定称呼方式，**不**原样念出去（ASR 文本可能有错字） */
  question?: string;
  /**
   * 同上，主进程接线（main.ts / IPC.llmAsk）用的字段名：
   * `buildFallbackAnswer(reason, { lastUserLine: transcript.at(-1) })`。
   * 与 `question` 等价，同时给时以 `question` 优先。
   */
  lastUserLine?: string;
  /** 资料里确实有支撑的相邻主题，用于"承认边界 + 说相邻经验"这条降级路线 */
  neighborTopic?: string;
  /** 变体序号：同一原因连续触发时换一条，避免复读同一句 */
  variant?: number;
}

/** 归一化后的上下文，模板只依赖这两个值。 */
interface FallbackSlots {
  /** 指向"面试官刚问的那个点"的主语短语 */
  point: string;
  /** 有资料支撑的相邻主题，'' = 调用方没给 */
  topic: string;
}

function fallbackSlots(ctx: FallbackContext = {}): FallbackSlots {
  const asked = (ctx.question ?? ctx.lastUserLine ?? '').trim().length > 0;
  return {
    point: asked ? '您问的这个点' : '这块',
    topic: (ctx.neighborTopic ?? '').trim(),
  };
}

/**
 * 三种降级策略，每句都真的能照着念（30 秒内、第一人称、口语、禁 Markdown）：
 *   ① 承认边界 + 说出相邻的、有资料支撑的经验
 *   ② 明确标注这是推断的类比
 *   ③ 一句得体争取时间的话术，并把话题拉回有资料的邻域
 * low_score 给 2 条不同措辞 —— 面试官连续追问时同一句话重复两遍最刺耳。
 */
const FALLBACK_BUILDERS: Record<FallbackReason, ((s: FallbackSlots) => string)[]> = {
  // ① / ③ 没建索引：什么都搜不到，只能把话题拉回候选人真做过的事
  no_index: [
    (s) => `${s.point}我没有现成的准备，就不硬答了。我更想讲我实打实做过的项目，里面的细节我能讲清楚，您看从哪个方向展开更合适？`,
    (s) =>
      `${s.point}我得如实说，我这边没有现成答案。` +
      (s.topic
        ? `我先讲我最熟的「${s.topic}」那部分，您判断对不对得上，不对我马上换方向。`
        : '我先讲我最熟的那部分，您判断对不对得上，不对我马上换方向。'),
  ],
  // ① 承认边界 + 相邻经验 / ② 标注为推断的类比
  low_score: [
    (s) =>
      `${s.point}我确实没有直接做过，不硬套。` +
      (s.topic
        ? `最接近的是我做的「${s.topic}」，我可以讲讲那次的思路，您看跟这个问题对不对得上。`
        : '我可以讲讲我项目里最接近的那一段，您看跟这个问题对不对得上。'),
    (s) => `${s.point}我手上没有直接经验，就不编了。我只能给您一个推断性的类比，不一定对，方向偏了您随时打断我。`,
  ],
  // ③ 争取时间 + 拉回有资料的邻域（检索本身失败了，不敢指任何具体经验）
  no_evidence: [
    (s) => `${s.point}我想先确认一下您最关心的是哪个角度，是做法还是最后的效果？我按您在意的这边讲，避免答偏。`,
    (s) => `${s.point}我不确定您想听哪一层，先讲我最有把握的那部分，讲完您告诉我偏没偏，我再往您要的方向补。`,
  ],
};

/** 该原因下的全部措辞（顺序固定，便于按 variant 轮换）。 */
export function fallbackPhrasings(reason: FallbackReason, ctx: FallbackContext = {}): string[] {
  const builders = FALLBACK_BUILDERS[reason] ?? FALLBACK_BUILDERS.no_evidence;
  const slots = fallbackSlots(ctx);
  return builders.map((build) => build(slots));
}

/**
 * 分层兜底话术：**不调用 LLM**，直接返回可照念的中文。
 * 这是"检索没找到"时的最终答案——宁可少说，也不能让模型自己补。
 */
export function buildFallbackAnswer(reason: FallbackReason, ctx: FallbackContext = {}): string {
  const list = fallbackPhrasings(reason, ctx);
  const n = list.length;
  const v = Math.trunc(ctx.variant ?? 0);
  return list[((v % n) + n) % n];
}

// ---------- P1-6: prefix-cache prewarm ----------

/**
 * The prewarm request: system prompt byte-identical to real answer requests
 * (that's the whole point — DeepSeek caches the common token prefix), plus a
 * constant one-word user turn; max_tokens=1 upstream, reply discarded.
 */
export function buildPrewarmMessages(stablePrefix: string): ChatMessage[] {
  return [
    { role: 'system', content: stablePrefix },
    { role: 'user', content: 'ok' },
  ];
}

export interface AnswerPromptInput {
  /** the sentence to answer (segment/continuous) or the text to translate */
  question?: string;
  /** recent transcript lines, oldest first */
  recentTranscript: string[];
  mode: 'segment' | 'continuous' | 'free' | 'translate';
  /** free-form user question (mode === 'free') */
  freeQuestion?: string;
  /** reply language for segment/continuous/free (default chinese) */
  answerLang?: AnswerLang;
  /** prior Q&A turns for a coherent session (oldest first) */
  history?: ChatMessage[];
  /** resume slot (双槽资料); falls back to `background` */
  resume?: string;
  /** job-description slot (双槽资料) */
  jd?: string;
  /** legacy single-slot KB / global default — treated as resume material */
  background?: string;
  /** rolling interview memo (P1) — slow-changing block, its own message */
  memo?: string;
  /**
   * RAG 检索到的证据（2.0）。非空时作为独立 user 消息插在 memo 之后、
   * history 之前；缺省/空数组时输出与 2.0 之前的版本**逐字节一致**。
   */
  evidence?: RagEvidence[];
  /** 输出是否必须带 [S#] 来源编号（对应 rag.requireCitations） */
  requireCitations?: boolean;
}

/** Keep the most recent lines within the char budget (oldest dropped first). */
export function clampTranscript(lines: string[], maxChars = MAX_CONTEXT_CHARS): string[] {
  const out: string[] = [];
  let total = 0;
  for (let i = lines.length - 1; i >= 0; i--) {
    const len = lines[i].length + 1;
    if (total + len > maxChars) break;
    out.unshift(lines[i]);
    total += len;
  }
  return out;
}

/**
 * Translate a transcript line to Chinese (R: 翻译功能). Fixed target = 中文,
 * output only the translation. If the text is already Chinese the model
 * simply echoes it.
 */
export function buildTranslateMessages(text: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        '你是翻译引擎。把用户给的整段文本翻译成【简体中文】。只输出译文本身，不要加引号、不要解释、不要复述原文；若原文已是中文则原样返回。',
    },
    { role: 'user', content: text.trim() },
  ];
}

/** R5: screenshot Q&A — one multimodal user message for a vision model. */
export function buildVisionMessages(
  question: string,
  imageDataUrl: string,
  background?: string,
): ChatMessage[] {
  const bg = (background ?? '').trim();
  const sys =
    '你是会议助手。用户发来一张屏幕截图（通常是对方共享的 PPT/文档或一道题目）。用中文简明回答用户关于截图的问题；若是提问/题目，给出用户可以直接说的回答要点或解题思路。' +
    (bg
      ? `\n\n===== 本人资料与知识库（作答时优先采用） =====\n${bg.slice(0, MAX_BACKGROUND_CHARS)}\n===== 资料结束 =====`
      : '');
  return [
    {
      role: 'system',
      content: sys,
    },
    {
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: imageDataUrl } },
        { type: 'text', text: question.trim() || '解读这页内容的要点，并给出我应该怎么回应的建议。' },
      ],
    },
  ];
}

export function buildAnswerMessages(input: AnswerPromptInput): ChatMessage[] {
  if (input.mode === 'translate') {
    return buildTranslateMessages(input.question ?? '');
  }

  const lang: AnswerLang = input.answerLang ?? 'chinese';
  const context = clampTranscript(input.recentTranscript);
  const resume = (input.resume ?? '').trim() || (input.background ?? '').trim();
  const jd = (input.jd ?? '').trim();
  // 证据为空 → 返回 []，调用方展开后与原路径逐字节一致（老路径零影响）
  const evidenceMsgs = buildEvidenceMessages(input.evidence ?? [], {
    requireCitations: input.requireCitations ?? false,
  });

  // Free "随便问": raw pass-through — NO meeting-assistant persona, so identity
  // / "which model are you" questions get the model's truthful answer. The
  // transcript + KB are offered only as optional reference.
  if (input.mode === 'free') {
    const refs: string[] = [];
    if (resume) refs.push(`【本人资料（简历）】\n${resume.slice(0, MAX_BACKGROUND_CHARS)}`);
    if (jd) refs.push(`【岗位JD】\n${jd.slice(0, MAX_BACKGROUND_CHARS)}`);
    if (context.length) refs.push(`【最近的对话转录】\n${context.join('\n')}`);
    const msgs: ChatMessage[] = [];
    if (refs.length) {
      msgs.push({ role: 'system', content: `以下资料供参考（可用可不用）：\n\n${refs.join('\n\n')}` });
    }
    msgs.push(...evidenceMsgs);
    msgs.push(...(input.history ?? []));
    msgs.push({ role: 'user', content: (input.freeQuestion ?? '').trim() });
    return msgs;
  }

  // segment / continuous: teleprompter with the stable prefix
  const msgs: ChatMessage[] = [{ role: 'system', content: buildStablePrefix(resume, jd, lang) }];

  const memo = (input.memo ?? '').trim();
  if (memo) {
    // slow-changing block sits BETWEEN the stable prefix and the fast history,
    // so a memo refresh only invalidates the cache from this point on
    msgs.push({ role: 'user', content: `【面试进行备忘】（此前面试内容的滚动摘要，保持前后一致）\n${memo}` });
    msgs.push({ role: 'assistant', content: '收到，我会保持一致。' });
  }

  // 证据层：memo 之后、history 之前。每轮都变的东西必须排在最后几轮才失效的位置上，
  // 这样 system + memo 这段公共前缀仍然逐字节复用（前缀缓存命中）。
  msgs.push(...evidenceMsgs);

  msgs.push(...(input.history ?? []));

  const contextBlock = context.length
    ? `【最近的对话转录】\n${context.join('\n')}`
    : '【最近的对话转录】（暂无）';
  const q = (input.question ?? '').trim();
  const hint = q ? questionHint(classifyQuestion(q)) : '';
  const ask = q
    ? `面试官刚才说：\n“${q}”\n${hint ? hint + '\n' : ''}请直接给出我可以照着念的回答。`
    : '基于上面最近的转录，面试官最新的话需要我回应。请直接给出我可以照着念的回答。';

  msgs.push({ role: 'user', content: `${contextBlock}\n\n${ask}` });
  return msgs;
}
