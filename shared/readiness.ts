/**
 * 面试前自检（pre-flight readiness check）。
 *
 * 为什么需要它：面试前最贵的错误是"开始之后才发现某项没配好"——比如
 * ASR 后端指向一个本机不存在的本地运行时（实测踩过：新装实例被向导设成
 * `local-realtime`，而 FunASR 的 conda 环境并不存在，结果一句话都转不出来）。
 * 这类问题在配置界面里看不出来，只有真去跑才会暴露，而那时面试已经开始了。
 *
 * 本模块把"能不能用"变成一组**可执行**的检查：每一项都给出
 * `ok / warn / fail` + **具体怎么修**，而不是只报一个红叉。
 *
 * 设计要点：
 *   * 纯逻辑 + 注入依赖（dependency injection），因此可以单测，
 *     不需要真的起 Electron、真的连网；
 *   * 检查项并行执行、各自带超时，避免某一项卡住导致整页不返回；
 *   * 检查只读，不修改任何配置——「自检」绝不能有副作用。
 */
import type {
  EmbeddingSettings,
  KbSlot,
  ReadinessCheck,
  ReadinessItem,
  ReadinessLevel,
  ReadinessReport,
} from './protocol';

/**
 * ASR 后端。用字面量联合而不是引用 protocol 的字段类型：
 * protocol 里它写在 `SettingsFile['asr']` 内联位置，没有独立导出名。
 */
export type AsrBackend = 'local' | 'cloud' | 'cloud-realtime' | 'local-realtime';

/** 本地 FunASR sidecar 的探测结果（由主进程注入） */
export interface LocalAsrProbe {
  /** 解释器路径是否存在 */
  interpreterFound: boolean;
  /** 解释器路径（用于提示信息） */
  interpreterPath: string;
  /** 端口是否已在监听（说明已有 sidecar 在跑） */
  portOpen: boolean;
}

export interface ReadinessDeps {
  /** 当前 ASR 后端 */
  asrBackend: AsrBackend;
  /** 本地 ASR 探测（后端为 local-realtime 时才需要） */
  probeLocalAsr: () => Promise<LocalAsrProbe>;
  /** LLM 连通性测试；返回 null = 没配 Key（不是错误，是要填） */
  testLlm: () => Promise<{ ok: boolean; label: string; ms?: number; error?: string } | null>;
  /**
   * 检索链路检查：返回索引概况 + 实测检索是否可用。
   * 这里只做**本地**判断（索引是否非空、模型是否一致），
   * 联网的 embedding 连通性由 testEmbedding 单独测，避免自检重复付费。
   */
  inspectRag: () => Promise<{
    enabled: boolean;
    chunkCount: number;
    sources: { name: string; slot: KbSlot; chunks: number }[];
    model?: string;
    embeddingModel: string;
    dim?: number;
    embeddingDim?: number;
    apiKeySet: boolean;
    /** 索引里各档位片段数（用于提示"有多少内容需要谨慎讲"） */
    byLevel: Record<'verified' | 'claimed' | 'unverified', number>;
  }>;
  /** embedding 连通性（可选：自检里点"深度检查"才跑，会消耗一次 API 调用） */
  testEmbedding?: () => Promise<{ ok: boolean; messageZh: string; dim?: number; latencyMs?: number }>;
  /** 是否要做联网的深度检查 */
  deep?: boolean;
}

/** 单项超时：任何一项卡住都不能让整页自检挂死 */
const TIMEOUT_MS = 12_000;

function withTimeout<T>(p: Promise<T>, ms = TIMEOUT_MS): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`检查超时（${ms / 1000}s 未返回）`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

/** 汇总项：把所有 level 归约成最差的那个。 */
export function worstLevel(items: ReadinessItem[]): ReadinessLevel {
  if (items.some((i) => i.level === 'fail')) return 'fail';
  if (items.some((i) => i.level === 'warn')) return 'warn';
  return 'ok';
}

export function summarise(items: ReadinessItem[]): { level: ReadinessLevel; summaryZh: string } {
  const fails = items.filter((i) => i.level === 'fail');
  const warns = items.filter((i) => i.level === 'warn');
  const level = worstLevel(items);
  if (level === 'fail') {
    return { level, summaryZh: `${fails.length} 项必须修好才能面试：${fails.map((f) => f.label).join('、')}` };
  }
  if (level === 'warn') {
    return { level, summaryZh: `可以进行，但有 ${warns.length} 处要留意：${warns.map((w) => w.label).join('、')}` };
  }
  return { level: 'ok', summaryZh: '全部就绪，可以开始面试。' };
}

/**
 * 检查 1：转写链路。
 *
 * 这是最容易踩坑的一项：向导可能把后端设成本地 FunASR，而本机根本没有
 * 对应的 Python 环境。**必须真的去探测**，而不是只看配置值。
 */
export async function checkAsr(deps: ReadinessDeps): Promise<ReadinessItem> {
  const base = { check: 'asr' as ReadinessCheck, label: '转写（面试官提问来源）' };
  try {
    if (deps.asrBackend === 'local-realtime' || deps.asrBackend === 'local') {
      const probe = await withTimeout(deps.probeLocalAsr());
      if (probe.portOpen) {
        return {
          ...base,
          level: 'ok',
          detailZh: `本地识别服务已在运行（${probe.interpreterPath}）`,
          fixZh: '',
        };
      }
      if (!probe.interpreterFound) {
        return {
          ...base,
          level: 'fail',
          detailZh: `后端是「本地 FunASR」，但没找到它的 Python 解释器：${probe.interpreterPath}`,
          fixZh:
            '两条路选一条：①（推荐，最快）设置 → STT 改为「阿里云实时」，填 DashScope Key；' +
            '② 装好 FunASR 环境（需要 Python + funasr 依赖 + 模型文件，约数 GB）。' +
            '不改的话面试时一句话都转不出来。',
        };
      }
      return {
        ...base,
        level: 'warn',
        detailZh: `解释器存在但本地识别服务还没起来：${probe.interpreterPath}`,
        fixZh: '开始采集时会自动拉起；建议现在先用「测试」按钮确认能起来，避免面试时才首次加载模型。',
      };
    }
    // 云识别：只要 Key 在就行（真正的连通性由 STT 设置页的测试按钮负责，
    // 这里不再发一次网络请求，避免"自检"自己产生网络副作用）
    return {
      ...base,
      level: 'ok',
      detailZh: `使用云识别（${deps.asrBackend}）`,
      fixZh: '',
    };
  } catch (e) {
    return {
      ...base,
      level: 'warn',
      detailZh: `无法确认本地识别服务状态：${(e as Error).message}`,
      fixZh: '若后端是云识别可忽略；若用本地识别，请到 STT 设置页点「测试」。',
    };
  }
}

/** 检查 2：大模型连通性。 */
export async function checkLlm(deps: ReadinessDeps): Promise<ReadinessItem> {
  const base = { check: 'llm' as ReadinessCheck, label: '大模型（生成回答）' };
  try {
    const r = await withTimeout(deps.testLlm());
    if (r === null) {
      return {
        ...base,
        level: 'fail',
        detailZh: '没有配置 API Key',
        fixZh: '设置 → LLM，填上 Key 并点「测试」。没有它就无法生成回答。',
      };
    }
    if (!r.ok) {
      return {
        ...base,
        level: 'fail',
        detailZh: `连接失败：${r.error ?? '未知原因'}`,
        fixZh: '设置 → LLM：核对 baseUrl（到 /v1 为止）、模型名与 Key；国内直连 DeepSeek 一般不需要代理。',
      };
    }
    return {
      ...base,
      level: 'ok',
      detailZh: `${r.label} 连接正常${r.ms ? `（${r.ms}ms）` : ''}`,
      fixZh: '',
    };
  } catch (e) {
    return {
      ...base,
      level: 'fail',
      detailZh: `连接失败：${(e as Error).message}`,
      fixZh: '检查网络与代理设置；DeepSeek 国内可直连。',
    };
  }
}

/** 检查 3：检索链路（索引 + embedding 配置一致性）。 */
export async function checkRag(deps: ReadinessDeps): Promise<ReadinessItem> {
  const base = { check: 'rag' as ReadinessCheck, label: '检索增强（依据你的资料作答）' };
  try {
    const r = await withTimeout(deps.inspectRag());
    if (!r.enabled) {
      return {
        ...base,
        level: 'warn',
        detailZh: '检索增强已关闭，回答不会引用你的项目资料',
        fixZh: '设置 →「检索增强 / RAG」把总开关打开；否则 AI 只能用通用知识回答，容易说空话。',
      };
    }
    if (!r.apiKeySet) {
      return {
        ...base,
        level: 'fail',
        detailZh: '没有配置 embedding API Key，无法检索',
        fixZh: '设置 →「检索增强 / RAG」→ API Key 填 DashScope Key（与实时转写共用同一个即可）→ 点「测试 embedding 连通性」。',
      };
    }
    if (r.chunkCount === 0) {
      return {
        ...base,
        level: 'fail',
        detailZh: '索引为空，还没有导入并索引任何资料',
        fixZh:
          '导入你的项目讲稿/简历/手册 → 在「检索增强 / RAG」点「重建索引」→ 等进度条走完。' +
          '注意：索引没建好时系统会自动退回"整份资料直塞"的老方式，检索不会生效。',
      };
    }
    // 模型/维度不一致 → 检索会静默失效（这是最难自查的一类故障）
    if (r.model && r.embeddingModel && r.model !== r.embeddingModel) {
      return {
        ...base,
        level: 'fail',
        detailZh: `索引是用「${r.model}」建的，但当前配置是「${r.embeddingModel}」——两者向量空间不同，检索会完全失效`,
        fixZh: '换 embedding 模型后必须重建索引：点「清空索引」再「重建索引」。',
      };
    }
    if (r.dim && r.embeddingDim && r.dim !== r.embeddingDim) {
      return {
        ...base,
        level: 'fail',
        detailZh: `索引维度 ${r.dim} 与当前 embedding 维度 ${r.embeddingDim} 不一致，检索会完全失效`,
        fixZh: '把维度改回建索引时的值（或清空索引后按新维度重建）。',
      };
    }
    const claimed = r.byLevel.claimed;
    const pct = r.chunkCount > 0 ? Math.round((claimed / r.chunkCount) * 100) : 0;
    const detail = `索引 ${r.chunkCount} 段（模型 ${r.model ?? '?'}）：已证实 ${r.byLevel.verified} / 待确认 ${claimed} / 未证实 ${r.byLevel.unverified}`;
    if (pct >= 70) {
      return {
        ...base,
        level: 'warn',
        detailZh: `${detail}。待确认占比 ${pct}% 偏高，模型会被约束得很谨慎，可能"什么都不敢讲"`,
        fixZh: '检查资料里是否大量出现数字指标；如果大部分其实有实测支撑，可考虑在资料里用【项目】标记说明。',
      };
    }
    return { ...base, level: 'ok', detailZh: detail, fixZh: '' };
  } catch (e) {
    return {
      ...base,
      level: 'fail',
      detailZh: `检查失败：${(e as Error).message}`,
      fixZh: '重试一次；若持续失败，把这条信息发给我。',
    };
  }
}

/** 检查 4：资料导入情况。 */
export async function checkMaterial(deps: ReadinessDeps): Promise<ReadinessItem> {
  const base = { check: 'material' as ReadinessCheck, label: '资料' };
  try {
    const r = await withTimeout(deps.inspectRag());
    if (r.sources.length === 0) {
      return {
        ...base,
        level: 'fail',
        detailZh: '还没有索引任何资料',
        fixZh: '导入你的项目讲稿（建议先用带【项目】/【边界】标注的那份），然后重建索引。',
      };
    }
    const list = r.sources.map((s) => `${s.name}（${s.slot === 'jd' ? 'JD' : '项目资料'}，${s.chunks} 段）`).join('；');
    const hasJd = r.sources.some((s) => s.slot === 'jd');
    if (!hasJd) {
      return {
        ...base,
        level: 'warn',
        detailZh: `已索引：${list}。但没有岗位 JD`,
        fixZh: '把目标岗位 JD 导入到 JD 槽位，回答才能"往岗位上贴"；没有明确目标岗位可以忽略。',
      };
    }
    return { ...base, level: 'ok', detailZh: `已索引：${list}`, fixZh: '' };
  } catch (e) {
    return { ...base, level: 'warn', detailZh: `无法读取资料状态：${(e as Error).message}`, fixZh: '' };
  }
}

/** 检查 5：embedding 连通性（仅在深度检查时联网）。 */
export async function checkEmbeddingDeep(deps: ReadinessDeps): Promise<ReadinessItem> {
  const base = { check: 'embedding' as ReadinessCheck, label: 'Embedding 连通性（深度检查）' };
  if (!deps.deep || !deps.testEmbedding) {
    return {
      ...base,
      level: 'ok',
      detailZh: '跳过（快速自检不发起网络请求）',
      fixZh: '',
    };
  }
  try {
    const r = await withTimeout(deps.testEmbedding());
    return r.ok
      ? { ...base, level: 'ok', detailZh: `可用：维度 ${r.dim}，${r.latencyMs}ms`, fixZh: '' }
      : { ...base, level: 'fail', detailZh: `不可用：${r.messageZh}`, fixZh: '核对 baseUrl、模型名与 Key；百炼端点不需要 WorkspaceId。' };
  } catch (e) {
    return { ...base, level: 'fail', detailZh: `不可用：${(e as Error).message}`, fixZh: '检查网络；百炼国内可直连。' };
  }
}

/**
 * 跑完整自检。所有检查**并行**执行，单项失败不影响其它项。
 * 顺序固定，保证界面稳定；汇总结果取最差档位。
 */
export async function runReadiness(deps: ReadinessDeps): Promise<ReadinessReport> {
  const items = await Promise.all([
    checkAsr(deps),
    checkLlm(deps),
    checkRag(deps),
    checkMaterial(deps),
    checkEmbeddingDeep(deps),
  ]);
  const { level, summaryZh } = summarise(items);
  return { level, summaryZh, items, checkedAt: Date.now(), deep: !!deps.deep };
}
