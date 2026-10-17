/**
 * RAG 提示词层测试（MeetingCopilot 2.0）。
 *
 * 守住四件事：
 *   1. **前缀缓存是钱**：evidence 为空/undefined 时，buildAnswerMessages 的输出
 *      必须与 2.0 之前的实现逐字节一致（用内联"参考实现"逐行比对，而不是肉眼）；
 *   2. 证据消息的位置：memo 之后、history 之前（每轮都变的东西不能放前面）；
 *   3. 事实档位约束真的写进了 prompt（claimed 数字原样引用、unverified 只能讲计划）；
 *   4. 三种兜底话术：中文、第一人称、禁 Markdown、绝不暴露 AI / 知识库的存在。
 */
import { describe, expect, it } from 'vitest';
import type { ChatMessage } from '../electron/llm/adapter';
import type { RagEvidence } from '../shared/protocol';
import { FACT_LEVEL_LABEL, FACT_LEVEL_RULE } from '../electron/rag/factlevel';
import {
  MAX_CONTEXT_CHARS,
  buildAnswerMessages,
  buildEvidenceMessages,
  buildFallbackAnswer,
  buildStablePrefix,
  buildTranslateMessages,
  clampTranscript,
  classifyQuestion,
  fallbackPhrasings,
  questionHint,
  type AnswerPromptInput,
  type FallbackReason,
} from '../electron/llm/prompts';

/** 固定输入的证据构造器：默认已验证档位，便于用例只覆盖关心的字段。 */
function mkEvidence(partial: Partial<RagEvidence> & { ref: string; text: string }): RagEvidence {
  return {
    chunkId: `chunk-${partial.ref}`,
    source: '讲稿.md',
    heading: '第3部分',
    factLevel: 'verified',
    score: 0.82,
    keywordBonus: 0.1,
    ...partial,
  };
}

const EVIDENCE_FIXTURE: RagEvidence[] = [
  mkEvidence({ ref: 'S1', text: '我负责海洋遥感多智能体知识服务平台的检索链路。' }),
  mkEvidence({
    ref: 'S2',
    text: '简历口径：检索准确率 91.3%，500 条测试样本。',
    factLevel: 'claimed',
    heading: '第4部分',
  }),
  mkEvidence({
    ref: 'S3',
    text: '预计下一版把首字延迟降到 300ms 以内。',
    factLevel: 'unverified',
    source: '面试讲稿.md',
  }),
];

// ---------------------------------------------------------------------------
// 参考实现：2.0 之前 buildAnswerMessages 的逐行副本（不含 evidence 层）。
// 它的唯一用途是证明"无证据时逐字节不变"。
// ---------------------------------------------------------------------------
function referenceAnswerMessages(input: AnswerPromptInput): ChatMessage[] {
  if (input.mode === 'translate') {
    return buildTranslateMessages(input.question ?? '');
  }

  const lang = input.answerLang ?? 'chinese';
  const context = clampTranscript(input.recentTranscript);
  const resume = (input.resume ?? '').trim() || (input.background ?? '').trim();
  const jd = (input.jd ?? '').trim();

  if (input.mode === 'free') {
    const refs: string[] = [];
    if (resume) refs.push(`【本人资料（简历）】\n${resume.slice(0, 8000)}`);
    if (jd) refs.push(`【岗位JD】\n${jd.slice(0, 8000)}`);
    if (context.length) refs.push(`【最近的对话转录】\n${context.join('\n')}`);
    const msgs: ChatMessage[] = [];
    if (refs.length) {
      msgs.push({ role: 'system', content: `以下资料供参考（可用可不用）：\n\n${refs.join('\n\n')}` });
    }
    msgs.push(...(input.history ?? []));
    msgs.push({ role: 'user', content: (input.freeQuestion ?? '').trim() });
    return msgs;
  }

  const msgs: ChatMessage[] = [{ role: 'system', content: buildStablePrefix(resume, jd, lang) }];

  const memo = (input.memo ?? '').trim();
  if (memo) {
    msgs.push({ role: 'user', content: `【面试进行备忘】（此前面试内容的滚动摘要，保持前后一致）\n${memo}` });
    msgs.push({ role: 'assistant', content: '收到，我会保持一致。' });
  }

  msgs.push(...(input.history ?? []));

  const contextBlock = context.length
    ? `【最近的对话转录】\n${context.join('\n')}`
    : `【最近的对话转录】（暂无）`;
  const q = (input.question ?? '').trim();
  const hint = q ? questionHint(classifyQuestion(q)) : '';
  const ask = q
    ? `面试官刚才说：\n“${q}”\n${hint ? hint + '\n' : ''}请直接给出我可以照着念的回答。`
    : '基于上面最近的转录，面试官最新的话需要我回应。请直接给出我可以照着念的回答。';

  msgs.push({ role: 'user', content: `${contextBlock}\n\n${ask}` });
  return msgs;
}

const HISTORY: ChatMessage[] = [
  { role: 'user', content: '上一个问题' },
  { role: 'assistant', content: '上一个回答' },
];

describe('buildAnswerMessages：无 evidence 时逐字节不变（前缀缓存安全）', () => {
  it('固定输入下的完整消息数组等于字面量（无 memo/history/资料）', () => {
    const msgs = buildAnswerMessages({
      mode: 'segment',
      question: '今天我们随便聊聊',
      recentTranscript: [],
    });
    expect(msgs).toEqual([
      { role: 'system', content: buildStablePrefix('', '', 'chinese') },
      {
        role: 'user',
        content: '【最近的对话转录】（暂无）\n\n面试官刚才说：\n“今天我们随便聊聊”\n请直接给出我可以照着念的回答。',
      },
    ]);
  });

  it('带 memo + history + 双槽资料时与参考实现逐字节一致', () => {
    const input: AnswerPromptInput = {
      mode: 'segment',
      question: '讲讲你的检索链路怎么做的',
      recentTranscript: ['面试官：先做个自我介绍吧', '我：好的'],
      memo: '【已问问题】自我介绍\n【我已声称的事实】三年经验',
      history: HISTORY,
      resume: '项目经历：海洋遥感知识服务平台，负责检索链路。',
      jd: '岗位职责：负责 RAG 检索与排序。',
      memo2: undefined,
    } as AnswerPromptInput;
    const actual = buildAnswerMessages(input);
    const expected = referenceAnswerMessages(input);
    expect(JSON.stringify(actual)).toBe(JSON.stringify(expected));
    expect(actual).toEqual(expected);
  });

  it('evidence 为 undefined 与 []) 三种输入（segment/continuous/free/translate）都与参考实现一致', () => {
    const cases: AnswerPromptInput[] = [
      { mode: 'segment', question: '你的核心优势是什么？', recentTranscript: ['寒暄', '你的核心优势是什么？'] },
      { mode: 'continuous', recentTranscript: ['刚才那句'], memo: '备忘' },
      { mode: 'continuous', question: '为什么离职？', recentTranscript: ['a'], history: HISTORY },
      {
        mode: 'segment',
        question: '未来规划',
        recentTranscript: ['x'.repeat(MAX_CONTEXT_CHARS + 100)],
        answerLang: 'english',
        resume: '简历内容',
      },
      { mode: 'free', freeQuestion: '总结一下刚才', recentTranscript: ['a'], background: '全局资料' },
      { mode: 'free', freeQuestion: '你是什么模型', recentTranscript: [] },
      { mode: 'translate', question: 'Hello world', recentTranscript: ['不应出现'] },
    ];
    for (const c of cases) {
      const label = `${c.mode}:${c.question ?? c.freeQuestion ?? ''}`;
      const expected = referenceAnswerMessages(c);
      expect(JSON.stringify(buildAnswerMessages(c)), label).toBe(JSON.stringify(expected));
      expect(JSON.stringify(buildAnswerMessages({ ...c, evidence: [] })), label).toBe(JSON.stringify(expected));
      expect(JSON.stringify(buildAnswerMessages({ ...c, evidence: undefined })), label).toBe(
        JSON.stringify(expected),
      );
      // 空证据绝不能留下任何痕迹
      expect(JSON.stringify(buildAnswerMessages({ ...c, evidence: [] }))).not.toContain('可用资料片段');
    }
  });

  it('证据不影响 stable prefix（系统消息与 evidence 无关）', () => {
    const base: AnswerPromptInput = {
      mode: 'segment',
      question: 'q',
      recentTranscript: ['a'],
      resume: '简历',
      jd: 'JD',
      memo: '备忘',
    };
    const without = buildAnswerMessages(base);
    const withEv = buildAnswerMessages({ ...base, evidence: EVIDENCE_FIXTURE, requireCitations: true });
    expect(withEv[0]).toEqual(without[0]);
    expect(withEv[0].content).toBe(buildStablePrefix('简历', 'JD', 'chinese'));
    expect(JSON.stringify(withEv[0])).not.toContain('可用资料片段');

    const a = buildStablePrefix('简历', 'JD', 'chinese');
    const b = buildStablePrefix('简历', 'JD', 'chinese');
    expect(a).toBe(b);
  });
});

describe('buildAnswerMessages：证据消息插在 memo 之后、history 之前', () => {
  it('segment 模式（有 memo）：system → memo → assistant → 证据 → assistant → history → 本次问题', () => {
    const msgs = buildAnswerMessages({
      mode: 'segment',
      question: '你的检索准确率怎么测的？',
      recentTranscript: ['面试官：你的检索准确率怎么测的？'],
      memo: '【已问问题】自我介绍',
      history: HISTORY,
      evidence: EVIDENCE_FIXTURE,
      requireCitations: true,
    });

    expect(msgs[0].role).toBe('system');
    expect(msgs[1].content).toContain('【面试进行备忘】');
    expect(msgs[2].role).toBe('assistant');
    expect(msgs[3].role).toBe('user');
    expect(msgs[3].content).toContain('【可用资料片段】');
    expect(msgs[4].role).toBe('assistant');
    expect(msgs[5]).toEqual(HISTORY[0]);
    expect(msgs[6]).toEqual(HISTORY[1]);
    expect(msgs[msgs.length - 1].content).toContain('请直接给出我可以照着念的回答。');

    // 关键的顺序断言：证据下标 > memo 确认，且 < history 首条
    const evIndex = msgs.findIndex((m) => typeof m.content === 'string' && m.content.includes('【可用资料片段】'));
    const memoIndex = msgs.findIndex((m) => typeof m.content === 'string' && m.content.includes('【面试进行备忘】'));
    expect(evIndex).toBeGreaterThan(memoIndex);
    expect(evIndex).toBeLessThan(msgs.findIndex((m) => m === HISTORY[0]));
  });

  it('segment 模式（无 memo）：证据紧跟在系统消息之后', () => {
    const msgs = buildAnswerMessages({
      mode: 'segment',
      question: 'q',
      recentTranscript: [],
      evidence: EVIDENCE_FIXTURE,
    });
    expect(msgs[0].role).toBe('system');
    expect(msgs[1].role).toBe('user');
    expect(msgs[1].content).toContain('【可用资料片段】');
    expect(msgs[2].role).toBe('assistant');
  });

  it('free 模式同样插在 history 之前（不破坏自由问答的裸问句收尾）', () => {
    const msgs = buildAnswerMessages({
      mode: 'free',
      freeQuestion: '总结一下',
      recentTranscript: [],
      history: HISTORY,
      evidence: EVIDENCE_FIXTURE,
    });
    expect(msgs[msgs.length - 1].content).toBe('总结一下');
    const evIndex = msgs.findIndex((m) => typeof m.content === 'string' && m.content.includes('【可用资料片段】'));
    expect(evIndex).toBeGreaterThan(-1);
    expect(evIndex).toBeLessThan(msgs.findIndex((m) => m === HISTORY[0]));
  });

  it('translate 模式绝不携带证据（与 resume/JD 一致的隔离）', () => {
    const msgs = buildAnswerMessages({
      mode: 'translate',
      question: 'Hello',
      recentTranscript: [],
      evidence: EVIDENCE_FIXTURE,
      requireCitations: true,
    });
    expect(JSON.stringify(msgs)).not.toContain('可用资料片段');
    expect(JSON.stringify(msgs)).not.toContain('检索链路');
  });

  it('空数组 / 空白正文不产生任何消息（老路径零影响）', () => {
    expect(buildEvidenceMessages([], { requireCitations: true })).toEqual([]);
    expect(buildEvidenceMessages([mkEvidence({ ref: 'S1', text: '   ' })], { requireCitations: true })).toEqual([]);
  });
});

describe('buildEvidenceMessages：编号、来源、档位与硬性约束', () => {
  const block = () => buildEvidenceMessages(EVIDENCE_FIXTURE, { requireCitations: false })[0].content as string;

  it('每条片段标注来源与档位：[S1]（来源：讲稿.md · 第3部分 · 已证实）正文', () => {
    expect(block()).toContain('[S1]（来源：讲稿.md · 第3部分 · 已证实）我负责海洋遥感多智能体知识服务平台的检索链路。');
    expect(block()).toContain('[S2]（来源：讲稿.md · 第4部分 · 目标值/待确认）');
    expect(block()).toContain('[S3]（来源：面试讲稿.md · 第3部分 · 未证实）');
  });

  it('每个出现过的档位都附上 FACT_LEVEL_RULE，并列出属于它的编号', () => {
    const text = block();
    for (const lv of ['verified', 'claimed', 'unverified'] as const) {
      expect(text).toContain(FACT_LEVEL_LABEL[lv]);
      expect(text).toContain(FACT_LEVEL_RULE[lv]);
    }
    expect(text).toContain(`· ${FACT_LEVEL_LABEL.verified}（[S1]）：${FACT_LEVEL_RULE.verified}`);
    expect(text).toContain(`· ${FACT_LEVEL_LABEL.claimed}（[S2]）：${FACT_LEVEL_RULE.claimed}`);
    expect(text).toContain(`· ${FACT_LEVEL_LABEL.unverified}（[S3]）：${FACT_LEVEL_RULE.unverified}`);
  });

  it('claimed 数字必须原样引用：禁止换算/取整/外推/四舍五入', () => {
    const text = block();
    expect(text).toContain('原样引用');
    expect(text).toContain('禁止换算');
    expect(text).toContain('取整');
    expect(text).toContain('外推');
    expect(text).toContain('四舍五入');
    expect(text).toContain(FACT_LEVEL_LABEL.claimed);
  });

  it('unverified 只能讲成计划/预期，不能讲成已完成', () => {
    const text = block();
    expect(text).toContain('只能讲成计划或预期');
    expect(text).toContain('已经完成');
    expect(text).toContain('已经上线');
  });

  it('硬性约束用可执行措辞：无来源的事实句不出现、资料里没有的一律不要补', () => {
    const text = block();
    expect(text).toContain('每一句事实性陈述都必须能指到上面某个编号');
    expect(text).toContain('指不到来源的事实句，一句都不要写出来');
    expect(text).toContain('资料里没有的能力、公司、项目、数字，一律不要补');
    // 不许出现"不要编造"这类可以被理解成"编得像真的就行"的空话
    expect(text).not.toContain('不要编造');
  });

  it('requireCitations=true 才要求输出带 [S#]，false 不带', () => {
    const withCite = buildEvidenceMessages(EVIDENCE_FIXTURE, { requireCitations: true })[0].content as string;
    expect(withCite).toContain('输出里每处事实都要带来源编号');
    expect(withCite).toContain('[S1][S2]');

    const noCite = buildEvidenceMessages(EVIDENCE_FIXTURE, { requireCitations: false })[0].content as string;
    expect(noCite).not.toContain('输出里每处事实都要带来源编号');
  });

  it('ref 缺失时用序号兜底，且片段顺序不变', () => {
    const text = buildEvidenceMessages(
      [mkEvidence({ ref: '', text: '第一段' }), mkEvidence({ ref: '  ', text: '第二段' })],
      { requireCitations: false },
    )[0].content as string;
    expect(text).toContain('[S1]（来源：讲稿.md · 第3部分 · 已证实）第一段');
    expect(text).toContain('[S2]（来源：讲稿.md · 第3部分 · 已证实）第二段');
    expect(text.indexOf('第一段')).toBeLessThan(text.indexOf('第二段'));
  });

  it('motivating 说明：注入形式是一条 user 消息 + assistant 确认', () => {
    const msgs = buildEvidenceMessages(EVIDENCE_FIXTURE, { requireCitations: false });
    expect(msgs.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(String(msgs[0].content)).toContain('【硬性约束】');
    expect(String(msgs[1].content)).toContain('片段里没有的不补');
  });
});

describe('buildFallbackAnswer：三种原因的兜底话术（不调用 LLM）', () => {
  const REASONS: FallbackReason[] = ['no_index', 'low_score', 'no_evidence'];

  it('三种原因都返回非空的中文第一人称话术（30 秒内）', () => {
    for (const reason of REASONS) {
      const text = buildFallbackAnswer(reason, { question: '你会做实时语音吗？' });
      expect(text.length, reason).toBeGreaterThan(20);
      expect(text.length, reason).toBeLessThanOrEqual(160);
      expect(text, reason).toContain('我');
      expect(text, reason).toMatch(/[\u4e00-\u9fa5]/);
    }
  });

  it('禁 Markdown、无换行、不暴露 AI 存在（不含"知识库/检索/索引"字眼）', () => {
    // neighborTopic 是用户资料里的真实主题（可能本身就叫"检索链路"），
    // 这里用中性主题，检查的是**话术模板自己的措辞**有没有暴露 AI。
    for (const reason of REASONS) {
      for (const text of fallbackPhrasings(reason, { question: 'q', neighborTopic: '海洋遥感知识服务平台' })) {
        expect(text, reason).not.toContain('\n');
        expect(text, reason).not.toContain('#');
        expect(text, reason).not.toContain('*');
        expect(text, reason).not.toContain('`');
        expect(text, reason).not.toContain('_');
        expect(text, reason).not.toContain('-');
        expect(text, reason).not.toMatch(/^\s*(?:[-+•]|\d+[.、)])\s/);
        expect(text, reason).not.toContain('知识库');
        expect(text, reason).not.toContain('检索');
        expect(text, reason).not.toContain('索引');
        expect(text, reason).not.toContain('根据我的知识');
        expect(text, reason).not.toContain('检索结果显示');
        expect(text, reason).not.toContain('模型');
      }
    }
  });

  it('low_score 给 2 条不同措辞，并按 variant 轮换（避免连续重复同一句）', () => {
    const list = fallbackPhrasings('low_score', {});
    expect(list.length).toBeGreaterThanOrEqual(2);
    expect(new Set(list).size).toBe(list.length);
    const v0 = buildFallbackAnswer('low_score', { variant: 0 });
    const v1 = buildFallbackAnswer('low_score', { variant: 1 });
    expect(v0).not.toBe(v1);
    expect(buildFallbackAnswer('low_score', { variant: 2 })).toBe(v0);
    expect(buildFallbackAnswer('low_score', { variant: -1 })).toBe(list[list.length - 1]);
  });

  it('三种降级策略都能落地：承认边界、标注为推断的类比、争取时间并拉回邻域', () => {
    expect(buildFallbackAnswer('low_score', { question: 'q', variant: 0 })).toContain('没有直接做过');
    expect(buildFallbackAnswer('low_score', { question: 'q', variant: 1 })).toContain('推断性的类比');
    expect(buildFallbackAnswer('no_evidence', { question: 'q', variant: 0 })).toContain('先确认一下您最关心的');
    expect(buildFallbackAnswer('no_index', { question: 'q', variant: 0 })).toContain('我没有现成的准备');
  });

  it('neighborTopic 有则点名相邻、有资料支撑的经验', () => {
    const text = buildFallbackAnswer('low_score', { question: 'q', neighborTopic: '海洋遥感知识服务平台' });
    expect(text).toContain('海洋遥感知识服务平台');
    expect(text).toContain('最接近的是我做的');

    const noTopic = buildFallbackAnswer('low_score', { question: 'q' });
    expect(noTopic).not.toContain('最接近的是我做的「');
    expect(noTopic).toContain('我项目里最接近的那一段');
  });

  it('question 只影响称呼，不把 ASR 原文念出去；且输出确定（可复现）', () => {
    const withQ = buildFallbackAnswer('no_evidence', { question: '这句话来自语音识别可能有错字' });
    expect(withQ).not.toContain('这句话来自语音识别可能有错字');
    expect(withQ.startsWith('您问的这个点')).toBe(true);

    const noQ = buildFallbackAnswer('no_evidence', {});
    expect(noQ.startsWith('这块')).toBe(true);

    expect(buildFallbackAnswer('no_index', {})).toBe(buildFallbackAnswer('no_index', {}));
    expect(buildFallbackAnswer('no_index', { variant: 5 })).toBe(buildFallbackAnswer('no_index', { variant: 1 }));
  });

  it('lastUserLine 是主进程接线的字段名（与 question 等价）', () => {
    // main.ts: buildFallbackAnswer(reason, { lastUserLine: transcript.at(-1) })
    expect(buildFallbackAnswer('low_score', { lastUserLine: '你会做实时语音吗？' })).toBe(
      buildFallbackAnswer('low_score', { question: '你会做实时语音吗？' }),
    );
    expect(buildFallbackAnswer('low_score', { lastUserLine: '你会做实时语音吗？' }).startsWith('您问的这个点')).toBe(
      true,
    );
    expect(buildFallbackAnswer('low_score', { lastUserLine: '  ' }).startsWith('这块')).toBe(true);
  });
});
