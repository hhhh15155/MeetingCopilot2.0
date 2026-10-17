/**
 * 事实边界标注（MeetingCopilot 2.0 的核心防瞎编机制）。
 *
 * 为什么需要它：检索只能决定"哪些句子有可能被说出来"，它**无法判断真假**。
 * 用户的求职资料里存在自承认的无仓库支撑数字，例如
 *   「简历里的『500 条 / Accuracy 91.3% / Macro-F1 0.89 / RAG 88.7%』
 *     在仓库中没有支撑，真实 baseline 仅 11 条样本、Accuracy 81.8%」
 * 而"500 条 / 91.3%"与"11 条 / 81.8%"在向量空间里语义高度接近，
 * 纯检索区分不了。若不加标注直接喂进模型，AI 会以用户的名义把包装值
 * 讲成既成事实——这是最危险的一类幻觉。
 *
 * 因此每个片段入库时都打上一个档位（FactLevel），检索时随片段一起注入，
 * 由 prompt 层强制约束模型如何使用：
 *
 *   verified   已证实：能指到仓库文件/模块/架构决策 → 可正常陈述
 *   claimed    目标值/包装值：简历写了但无仓库支撑 → 只能如实说明是目标/包装
 *   unverified 未证实：资料里的"预计/大概/规划" → 只能当计划讲
 *
 * 判定是**确定性规则**（无 LLM），因为：
 *   1. 索引是一次性离线任务，结果要可复现、可人工核对；
 *   2. 面试前无法容忍"标注器自己也在猜"。
 * 用户的资料里本来就带人工标注（【项目】/【通用】/【边界】），
 * 这些**显式标记优先级最高**，规则只在没有显式标记时兜底。
 */
import type { FactLevel } from '../../shared/protocol';

/** 档位的中文名，用于 prompt 与界面。 */
export const FACT_LEVEL_LABEL: Record<FactLevel, string> = {
  verified: '已证实',
  claimed: '目标值/待确认',
  unverified: '未证实',
};

/** 注入 prompt 时给模型的处理指令。 */
export const FACT_LEVEL_RULE: Record<FactLevel, string> = {
  verified: '可正常陈述，作为你的真实经历讲。',
  claimed: '这是我简历上的目标值/包装说法，缺少实测支撑。绝不能说成既成事实；被追问时要如实说明这是目标或口径。',
  unverified: '资料里只是规划/估计，没落地。只能作为计划或期望来讲，不能当成已完成。',
};

/**
 * 显式标记（最高优先级）。用户自己的资料里带人工分级，
 * 这类信息比任何正则都可靠，必须优先采纳。
 */
const EXPLICIT_BOUNDARY = [
  /【\s*边界\s*】/,
  /【\s*红线\s*】/,
  /\b边界\b/,
  /没有支撑/,
  /无(?:仓库)?支撑/,
  /无法(?:复现|复算|证实)/,
  /不能(?:声称|说成|当作)/,
  /如\s*实(?:说明|回答|说)/,
  /真实\s*(?:baseline|基线|数值|数据)/,
];

/** 明确的"已证实"人工标记 */
const EXPLICIT_VERIFIED = [
  /【\s*项目\s*】/,
  /【\s*已(?:证实|验证|完成)\s*】/,
  /【\s*代码\s*】/,
];

/** 营销/包装口吻 → 倾向 claimed（简历数字几乎必然走这里） */
const CLAIM_MARKERS = [
  /简历/,
  /包装/,
  /口径/,
  /对外(?:说|称|表述)/,
  /目标值/,
  /预期(?:值|指标)/,
  /做到了\s*\d/i,
];

/** 规划/估计口吻 → unverified */
const UNVERIFIED_MARKERS = [
  /预计/,
  /预计将/,
  /规划/,
  /计划/,
  /打算/,
  /大概/,
  /大约/,
  /估计/,
  /后续(?:会|将|计划)/,
  /未来(?:会|将)/,
  /设想/,
  /待(?:办|补|验证|确认)/,
  /TODO/i,
];

/**
 * 可指向仓库/代码的**弱**证据。
 *
 * ⚠️ 这里**绝不能出现**「模块 / 接口 / 函数 / 方法 / 类 / 实现了」这类词。
 * 它们恰恰是简历里密度最高的措辞，一旦算作证据，就会出现
 *   「我负责的模块准确率 91.3%」→ verified → prompt 让模型"可正常陈述"
 * 这种最危险的漏判（对抗性验证实测复现）。
 * 真正能证明数字的证据必须是**可指认的产物**：文件名、路径、commit 等。
 */
const REPO_HINTS = [
  /[\w./-]+\.(ts|tsx|js|jsx|py|rs|java|go|md|json|yml|yaml)\b/,
  /\b(?:commit|git|diff|PR|branch)\b/i,
  /\b(?:src|electron|shared|test|tools)\//,
  /行号|行\s*\d+/,
];

/**
 * **强**证据：只有这类措辞才允许把"带数字/指标的句子"判为已证实。
 * 语义 = "我实际测过/跑过/可复算"，而不是"我写了个模块"。
 */
const STRONG_EVIDENCE = [
  /实测/,
  /跑通/,
  /(?:验证|校验)过/,
  /可复算|可复现|可重现/,
  /复算|复现/,
  /压测|基准测试|benchmark/i,
  /测试结果|评测结果|实验数据/,
];

/**
 * 句中出现"数字或质量指标"即为高危句，必须有强证据才放行。
 *
 * 覆盖对抗性验证里逃逸的两类裸数字：
 *   - 裸小数指标：`Macro-F1 达到 0.89`、`准确率 81.8`（F1/precision/recall 常写 0–1 小数）
 *   - 裸整数规模/性能：`QPS 从 300 提到了 1200`
 * 这类数字是幻觉的高危区：模型极爱照抄，也极爱四舍五入与外推。
 */
/**
 * 句中是否含**高危数字**。
 *
 * ⚠️ 关键设计取舍：不是"有数字就算风险"。
 * 实测在真实讲稿上，若把任意数字都判为高危，claimed 占比会达到 **91.7%**，
 * 简历手册更是 **100%**——徽章满屏橙色，用户立刻脱敏，反而让真正危险的
 * 简历包装数字淹没在噪声里（防瞎编机制自我失效）。
 *
 * 因此只把两类当作高危：
 *   1. **质量指标**（准确率/F1/召回/QPS/延迟/吞吐…）：简历包装的首选目标，
 *      且必须能指认到实测依据，否则一律 claimed。
 *   2. **带计量单位的数字**（500 条 / 88.7% / 3 倍 / 800ms）：典型包装量化。
 * 架构描述里的普通计数（"三个 Agent"、"K 取 60"、代码里的常量）不算风险。
 */
const NUMERIC_CLAIM_RE = new RegExp(
  [
    // 1) 质量/性能指标词后 0–6 字内出现数字
    String.raw`(?:准确率|精度|召回|查准|查全|f1|f-1|macro[- ]?f1|auc|map|ndcg|precision|recall|accuracy|iou|bleu|rouge|p95|p99|分位|qps|tps|rps|吞吐|耗时|延迟|并发|准确|正确率|错误率|通过率|覆盖率)\D{0,6}\d`,
    // 2) 带计量单位的数字（含"规模/样本/测试集"这类语境）
    String.raw`\d+(?:\.\d+)?\s*(?:%|％|倍|条|例|万个|万|千|k\b|ms|毫秒|秒|分钟|小时|个百分点)`,
    // 3) "规模/样本/测试集/数据集" 后的裸数字
    String.raw`(?:规模|样本|测试集|数据集|用例数|条数)\D{0,4}\d`,
    // 4) "从 X 提到 Y" 型对比（含裸整数）
    String.raw`从\s*\d+(?:\.\d+)?\s*(?:%|％)?\s*(?:提到|提升到|提高到|涨到|降到|到)`,
    // 5) 裸小数紧跟性能动词（0.xx 形态基本必然是模型指标）
    String.raw`(?:达到|提升到|提高到|做到|提升|提高|下降|降低|增长)\D{0,4}\d+\.\d+`,
    // 6) 裸整数紧跟性能动词
    String.raw`(?:提到|提升到|提高到|涨到|达到|做到)\s*\d{2,}`,
  ].join('|'),
  'i',
);

/** 兼容旧名（其他模块/测试可能引用）：判断是否含带单位指标。 */
const METRIC_RE = /(?:\d+(?:\.\d+)?\s*(?:%|％|倍|条|例|个|次|万|k|ms|毫秒|秒|s\b|分钟))/i;

/**
 * 剥离代码块与行内代码后再做档位判定。
 *
 * 代码里的数字（常量、阈值、数组长度）是**实现事实**，不是业绩声明；
 * 不剥离会把整份技术文档染成 claimed（实测该因素是过激标注的主要来源之一）。
 */
function stripCode(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/~~~[\s\S]*?~~~/g, ' ')
    .replace(/`[^`\n]*`/g, ' ');
}

/** 每段的切分粒度：按行/句，逐句判定后取最保守（风险最高）的档位。 */
function segments(text: string): string[] {
  return text
    .split(/(?<=[。！？；!?;\n])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** 风险排序：越高越保守，最终取整块里最高的那个。 */
const RISK: Record<FactLevel, number> = { verified: 0, unverified: 1, claimed: 2 };

/** 判定单句的档位。决策顺序即保守程度顺序。 */
export function classifySentence(sentence: string): FactLevel {
  // 代码块/行内代码里的数字是实现事实，不参与业绩风险判定
  const s = stripCode(sentence).trim();
  if (!s) return 'verified';

  // 1) 显式人工标记（【边界】/自承认无支撑）——最高优先级
  if (EXPLICIT_BOUNDARY.some((re) => re.test(s))) return 'claimed';

  const numericClaim = NUMERIC_CLAIM_RE.test(s) || METRIC_RE.test(s);
  const looksClaim = CLAIM_MARKERS.some((re) => re.test(s));
  const looksPlanned = UNVERIFIED_MARKERS.some((re) => re.test(s));
  const strongEvidence = STRONG_EVIDENCE.some((re) => re.test(s));
  const repoHint = REPO_HINTS.some((re) => re.test(s));

  // 2) 规划/估计口吻 → unverified，**优先于强证据**。
  //    "后续计划做压测" 里虽然出现了"压测"，但那是**将来时**——
  //    若因为出现强证据词就判 verified，模型会把计划讲成已完成。
  if (looksPlanned) return 'unverified';

  // 3) 包装口吻（简历/口径/目标值）→ claimed
  if (looksClaim) return 'claimed';

  // 4) ★核心防线★ 句中有任何数字或质量指标，必须有"实测/跑通/可复算"级
  //    强证据才允许算已证实；否则一律 claimed。
  //    注意顺序：这一步必须在"有仓库证据→verified"之前，
  //    否则「我负责的模块准确率 91.3%」会因为出现"模块"而被放行。
  if (numericClaim && !strongEvidence) return 'claimed';

  // 5) 强证据（实测/跑通/可复算）→ verified
  if (strongEvidence) return 'verified';

  // 6) 无数字的事实陈述：显式【项目】标记，或能指认到文件/路径 → verified
  if (EXPLICIT_VERIFIED.some((re) => re.test(s)) || repoHint) return 'verified';

  // 7) 兜底：无数字、无证据的普通陈述句（如"我负责数据清洗"）视为已证实。
  //    这里刻意不返回 claimed——否则整块资料都会变橙色徽章，
  //    用户会对警示色脱敏，反而削弱真正的风险提示。
  return 'verified';
}

/**
 * 判定整块片段的档位 = 块内最高风险档位。
 * 保守优先：一块里只要有一句是包装值，整块都按包装值约束，
 * 宁可让模型说得谨慎，也不能让它把某个数字当事实讲出去。
 */
export function classifyChunk(text: string): FactLevel {
  const segs = segments(text);
  if (segs.length === 0) return 'verified';
  let worst: FactLevel = 'verified';
  for (const s of segs) {
    const lv = classifySentence(s);
    if (RISK[lv] > RISK[worst]) worst = lv;
  }
  return worst;
}

/**
 * 统计一块里各档位的句子数，供界面显示"这段资料有多可信"。
 */
export function factLevelBreakdown(text: string): Record<FactLevel, number> {
  const out: Record<FactLevel, number> = { verified: 0, claimed: 0, unverified: 0 };
  for (const s of segments(text)) out[classifySentence(s)]++;
  return out;
}
