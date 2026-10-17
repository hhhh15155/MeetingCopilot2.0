/**
 * 引用覆盖率审计：对模型生成的回答做**确定性**核对，看哪些事实句没有出处。
 *
 * 为什么需要它：L1-L3 都在"尽量别让模型编"，但**编没编最终要能当场看出来**。
 * 只显示"参考了哪几段"还不够——用户看到的是整段回答，无法判断其中某个数字
 * 是不是有依据。这一层把核对结果直接摆在答案下面：
 *
 *   * 引用了哪些片段、有没有引用不存在的编号（幻觉来源）
 *   * 哪些**事实句**（含数字/公司/项目名）没有 `[S#]` 标注
 *
 * 刻意做成纯规则、不用 LLM：
 *   1. 面试时不需要多一次网络往返；
 *   2. "哪些句子带数字却没标来源"是客观事实，不该再让模型判断；
 *   3. 可单元测试、结果可复现。
 *
 * 注意力放在**高危句**：不含任何具体指称的泛化陈述（"我觉得沟通很重要"）
 * 不要求引用——否则满屏告警，用户很快就无视了。
 *
 * 类型 `CitationAudit` 的单一事实源在 shared/protocol.ts，本模块**单向**依赖它。
 */
import type { CitationAudit, RagEvidence } from './protocol';

/** 含数字的句子：这类是最需要出处的（模型最爱照抄也最爱外推） */
const HAS_NUMBER = /\d/;
/** 具体指称：公司/项目/仓库/技术名等，通常是可核对的事实 */
const HAS_PROPER_NOUN =
  /[A-Za-z][A-Za-z0-9+#./_-]{1,}|[\u4e00-\u9fff]{2,}(?:公司|平台|项目|系统|仓库|框架|实验室|研究院|大学)/;

/** 引用编号：`[S1]`、`【S1】`、`[S1][S2]`、`[S1, S2]`、`[S1 S2]`、小写 `[s1]` */
const CITE_RE = /[[【]\s*(s\d+(?:\s*[,，、]?\s*s\d+)*)\s*[\]】]/gi;

/**
 * 把一个回答切成句子。
 *
 * 按**行**切而不是只按句号：讲稿式的回答都是"一个要点一行"，
 * 逐行核对更贴近用户实际阅读的粒度。行内再按句末标点细分，
 * 但**不以逗号/顿号切**——中文长句的从句没有独立事实地位，
 * 切开会把"我做的 XX 项目，召回率 88.7%"拆成两句从而误报。
 */
export function splitAnswerSentences(answer: string): string[] {
  const out: string[] = [];
  for (const rawLine of answer.split(/\n+/)) {
    const line = rawLine.trim();
    if (!line) continue;
    // 去掉 Markdown 列表符号与加粗标记，保留正文
    const cleaned = line.replace(/^[-*•]\s*/, '').replace(/\*\*/g, '');
    for (const s of cleaned.split(/(?<=[。！？；!?;])/)) {
      const t = s.trim();
      if (t) out.push(t);
    }
  }
  return out;
}

/** 提取一句里的所有引用编号（大写归一化）。 */
export function parseCitations(sentence: string): string[] {
  const found: string[] = [];
  for (const m of sentence.matchAll(CITE_RE)) {
    for (const part of m[1].split(/[,，、\s]+/)) {
      const id = part.trim().toUpperCase();
      if (/^S\d+$/.test(id)) found.push(id);
    }
  }
  return found;
}

/**
 * 判断一句是否**需要**引用：含数字或具体指称。
 * 泛化陈述（"我比较善于沟通"）返回 false，不参与覆盖率统计。
 */
export function needsCitation(sentence: string): boolean {
  return HAS_NUMBER.test(sentence) || HAS_PROPER_NOUN.test(sentence);
}

/**
 * 主入口：核对回答与本次提供的证据是否对得上。
 *
 * @param answer 模型输出（流式结束后拿到时的**完整**文本才准确）
 * @param evidence 本次注入的证据列表（用于识别"引用不存在的编号"）
 */
export function auditCitations(answer: string, evidence: RagEvidence[]): CitationAudit {
  const provided = new Set(evidence.map((e) => e.ref.toUpperCase()));
  const sentences = splitAnswerSentences(answer);
  const citedSet = new Set<string>();
  const danglingSet = new Set<string>();
  const uncitedFacts: string[] = [];
  let factCount = 0;

  for (const s of sentences) {
    const cites = parseCitations(s);
    let valid = 0;
    for (const c of cites) {
      if (provided.has(c)) {
        citedSet.add(c);
        valid++;
      } else {
        danglingSet.add(c);
      }
    }
    if (needsCitation(s)) {
      factCount++;
      // 只有**有效**引用才算有出处：引用了不存在的编号等于没引用
      // （那正是"编造来源"的典型形态，绝不能因此被判为合规）。
      if (valid === 0) uncitedFacts.push(s);
    }
  }

  const sorted = [...citedSet].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  return {
    cited: sorted,
    dangling: [...danglingSet].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1))),
    uncitedFacts,
    factCount,
    coverage: factCount === 0 ? 1 : (factCount - uncitedFacts.length) / factCount,
  };
}
