/**
 * SDD contract: IPC protocol between main <-> renderer, and the settings schema.
 * Version this file; breaking changes bump PROTOCOL_VERSION.
 */
export const PROTOCOL_VERSION = 1;

import type { ProviderCapability, ProviderId } from './providerCatalog';
export type { ProviderCapability, ProviderId } from './providerCatalog';
import type { TrayRendererCommand } from './trayMenu';
export type { TrayCommand, TrayRendererCommand } from './trayMenu';

// ---------- Settings ----------

export type AsrLanguage = 'auto' | 'chinese' | 'english';
export type AnswerLang = 'chinese' | 'english';

/**
 * 推理档位，对应 OpenAI 兼容的 `reasoning_effort`。
 * `'none'` = 关掉思考模式，直接出正文——实时提词场景的正确选择。
 */
export type ReasoningEffort = 'none' | 'low' | 'high' | 'max';
/** who is speaking: the other party (system audio) vs the user (microphone) */
export type Speaker = 'them' | 'me';
/** answer-body font size (right pane only) */
export type FontScale = 'small' | 'medium' | 'large';
/** UI theme; 'system' follows prefers-color-scheme */
export type ThemeMode = 'dark' | 'light' | 'system';
/** UI display language (independent of answerLang, which steers the LLM) */
export type UiLang = 'zh' | 'en';
/** per-session material slots: resume vs job description */
/**
 * 资料槽位。
 *
 * 三槽而不是两槽：真实面试要同时带上**简历**（我是谁）、**项目资料**
 * （我做过什么，通常有多份讲稿/手册）和**岗位 JD**（往哪儿贴）。
 * 之前只有简历槽 + JD 槽，项目讲稿只能塞进简历槽，和真正的简历混在一起，
 * 检索时互相稀释。
 */
export type KbSlot = 'resume' | 'project' | 'jd';

/** 槽位显示顺序（界面、提示词、诊断都按这个顺序，避免各处不一致） */
export const KB_SLOTS: readonly KbSlot[] = ['resume', 'project', 'jd'];

/**
 * 一个槽位下的一份已导入文档。
 *
 * `id` 稳定标识这份文档（删除单份、逐份建索引用），不随数组下标变化。
 */
export interface KbDoc {
  id: string;
  /** 显示名（原始文件名） */
  name: string;
  /** 已解析出的纯文本 */
  text: string;
  /** 字节数，仅用于界面展示"这份资料多大" */
  size?: number;
  /** 导入时间戳 */
  addedAt: number;
}

// ============================================================================
// MeetingCopilot 2.0 — 接地气 RAG（外部 API embedding）
// ============================================================================
//
// 设计要点（借鉴 NexQ 的 RAG 思路，但大幅做减法）：
//   * embedding 走外部 HTTP API（阿里云百炼 text-embedding-v3 等），
//     不依赖本地 Ollama / Rust / 原生模块 —— 索引是一次性离线任务，
//     运行时每次提问只调一次 query embedding。
//   * 存储用 JSON + 进程内暴力余弦（11 万字 ≈ 300 chunk，毫秒级），
//     不引入 SQLite/FTS5/ANN 索引，避免污染零原生依赖的打包链。
//   * 中文不做"为混合而混合"的 FTS：向量为主，子串命中加分兜底。
//   * 检索的第一职责不是"找料"，而是产出**"没找到"信号**——这是防瞎编的地基。

/** 事实边界档位：决定 AI 能把这个片段当作多硬的事实来用。 */
export type FactLevel =
  /** 已证实：仓库/代码/可指的文件模块，可以正常陈述 */
  | 'verified'
  /** 目标值/包装值：简历上写了但无仓库支撑的数字（如"500 条、91.3%"） */
  | 'claimed'
  /** 未证实：讲稿里的"预计/大概/计划"，只能当规划说 */
  | 'unverified';

/**
 * 引用覆盖率审计结果。
 *
 * 为什么要做这一层：L1-L3 都在"尽量别让模型编"，但**编没编最终要能当场看出来**。
 * 只显示"参考了哪几段"还不够——用户看到的是整段回答，无法判断其中某个数字
 * 是不是有依据。UI 据此提示"哪些事实句没有出处"与"引用了不存在的编号"。
 *
 * 纯规则、不用 LLM：面试时不必多一次网络往返，且"带数字却没标来源"是客观事实，
 * 不该再让模型判断。实现见 shared/citationAudit.ts（该模块单向依赖本文件）。
 */
export interface CitationAudit {
  /** 回答里出现过的、且确实提供过的引用编号（去重升序），如 ['S1','S3'] */
  cited: string[];
  /** 引用了但本次从未提供过的编号（典型的"编造来源"） */
  dangling: string[];
  /** 含具体数字/指称、却没有引用编号的句子 */
  uncitedFacts: string[];
  /** 事实句总数（用于算覆盖率） */
  factCount: number;
  /**
   * 事实句里被引用的比例 0..1。`factCount === 0` 时为 1
   * （没有可核对的事实句，不应误报成"没引用"）。
   */
  coverage: number;
}

// ---------- 面试前自检（pre-flight readiness） ----------
//
// 实现见 shared/readiness.ts（该模块单向依赖本文件）。
// 放在协议里是为了让主进程与渲染进程共用同一套形状。

/** 单项检查的结果档位：fail 会让面试无法进行，warn 只是要留意 */
export type ReadinessLevel = 'ok' | 'warn' | 'fail';

/** 检查项标识（渲染进程据此决定要不要显示"去修"的跳转） */
export type ReadinessCheck = 'asr' | 'llm' | 'rag' | 'material' | 'embedding';

export interface ReadinessItem {
  check: ReadinessCheck;
  /** 中文短标签，如「转写（面试官提问来源）」 */
  label: string;
  level: ReadinessLevel;
  /** 现状说明（含关键数据，如耗时、片段数） */
  detailZh: string;
  /** **具体怎么修**；ok 时为空串。这一项是自检的核心价值：不能只报红叉 */
  fixZh: string;
}

export interface ReadinessReport {
  level: ReadinessLevel;
  /** 一句话结论，如「2 项必须修好才能面试：转写、检索增强」 */
  summaryZh: string;
  items: ReadinessItem[];
  checkedAt: number;
  /** 是否做了联网的深度检查 */
  deep: boolean;
}

/** 一个已索引的资料片段。 */
export interface RagChunk {
  id: string;
  /** 来源文件显示名 */
  source: string;
  slot: KbSlot;
  /** 片段在原文中的序号（0 基） */
  index: number;
  /** 原文所属标题路径，如 "第 3 部分 / 项目细节" */
  heading: string;
  text: string;
  /** 事实边界档位；入库时由标注规则判定 */
  factLevel: FactLevel;
  /** embedding 向量；未索引成功时为 undefined */
  vector?: number[];
}

/** 检索命中的一条证据（注入 prompt 的最小单元）。 */
export interface RagEvidence {
  /** 注入 prompt 时用的编号 [S1][S2]… */
  ref: string;
  chunkId: string;
  source: string;
  heading: string;
  text: string;
  factLevel: FactLevel;
  /** 原始余弦相似度 0..1（**不**用 RRF 归一化分数——阈值必须可解释） */
  score: number;
  /** 子串/专有名词精确命中加分 */
  keywordBonus: number;
}

/** 一次检索的完整结果，含"有没有找到"的负信号。 */
export interface RagRetrieval {
  evidence: RagEvidence[];
  /** 最高分片段的原始余弦；无命中时为 0 */
  topScore: number;
  /** true 表示证据不足，应走兜底话术而**不**交给 LLM 自由发挥 */
  lowConfidence: boolean;
  /** 检索耗时（毫秒），用于界面显示 */
  latencyMs: number;
}

export type RagIndexStatus = 'idle' | 'chunking' | 'embedding' | 'ready' | 'error';

/** 索引进度事件（主进程 -> 渲染进程）。 */
export interface RagIndexProgress {
  slot: KbSlot;
  status: RagIndexStatus;
  source: string;
  chunksTotal: number;
  chunksEmbedded: number;
  /** status==='error' 时的中文可读原因 */
  error?: string;
}

/** 知识库总览（供界面展示）。 */
export interface RagIndexInfo {
  chunkCount: number;
  /** 实际参与索引的来源文件名 */
  sources: { name: string; slot: KbSlot; chunks: number }[];
  /** 建立索引的时间戳 */
  indexedAt?: number;
  /** embedding 模型标识，换模型后需重建索引 */
  model?: string;
  dim?: number;
}

export interface EmbeddingSettings {
  providerId: string;
  baseUrl: string;
  model: string;
  /** 维度；0/undefined = 用服务端默认 */
  dim?: number;
  /** 建索引与检索共用的批大小 */
  batchSize?: number;
}

/** outcome of a provider connection test (Phase 3 runs them; the settings
 * schema stores the last result so the UI can show it after a restart) */
export type ProviderTestCode =
  | 'OK'
  | 'INVALID_KEY'
  | 'PERMISSION_DENIED'
  | 'INSUFFICIENT_BALANCE'
  | 'RATE_LIMITED'
  | 'MODEL_NOT_FOUND'
  | 'REGION_MISMATCH'
  | 'NETWORK_UNREACHABLE'
  | 'DNS_ERROR'
  | 'TLS_ERROR'
  | 'PROXY_ERROR'
  | 'TIMEOUT'
  | 'PROVIDER_ERROR'
  | 'UNKNOWN_ERROR';

/** last connection-test result for one provider slot */
export interface ProviderVerification {
  /** ISO-8601 */
  lastTestAt?: string;
  lastTestOk?: boolean;
  lastTestCode?: ProviderTestCode;
  latencyMs?: number;
}

/**
 * Which stored credential/verification slot a test belongs to. Mirrors the
 * settings layout: `asr.cloud` and `asr.realtime` are separate slots so
 * switching backends never clobbers the other one's key or test result.
 */
export type ProviderSlot = 'llm' | 'vision' | 'asr-cloud' | 'asr-realtime';

/**
 * renderer -> main: run ONE connection test. Explicit user action only — the
 * app never tests on startup, and every test costs the user a (minimal) API
 * call: 1 token, ~1.4 s of audio, or a 64x64 image.
 */
export interface ProviderTestRequest {
  /** catalog preset the user picked, when the choice came from the catalog */
  presetId?: string;
  capability: ProviderCapability;
  providerId: ProviderId;
  baseUrl: string;
  model: string;
  /**
   * Plaintext key to test BEFORE it is saved. renderer -> main only: it is
   * never persisted by this call, never logged, and never echoed back.
   */
  candidateApiKey?: string;
  /** test the key already stored in `slot` instead of supplying one */
  useStoredKey?: boolean;
  /** where to read the stored key from and where to record the verdict */
  slot?: ProviderSlot;
  /** vision only: '127.0.0.1:7897'-style local proxy */
  proxyUrl?: string;
  /**
   * ASR only: which bundled test clip to send and which language hint to pass.
   * Absent = main fills it in from `asr.language`.
   */
  language?: AsrLanguage;
}

/**
 * main -> renderer verdict. The messages are built by the unified mapper
 * (electron/providerErrors.ts), so the renderer never has to interpret a raw
 * provider error — which is also what keeps keys and Authorization headers out
 * of the UI.
 */
export interface ProviderTestResult {
  ok: boolean;
  code: ProviderTestCode;
  /** round-trip of the test call itself, present on success and on failure */
  latencyMs?: number;
  messageZh: string;
  messageEn: string;
  /** for the collapsible advanced detail, when the provider volunteered one */
  providerRequestId?: string;
  retryable: boolean;
}

/** the plan the user picked in the first-run wizard */
export type OnboardingPlan = 'recommended' | 'mimo-simple' | 'transcription-only' | 'advanced';

/** first-run wizard state (settings v2) */
export interface OnboardingState {
  /** wizard-state schema, independent of the settings-file version */
  schemaVersion: 1;
  /** false = the setup window owns startup; the main window never opens */
  completed: boolean;
  /** ISO-8601 timestamp of completion */
  completedAt?: string;
  /** 1-based step the user got to (for 保存并稍后继续) */
  lastStep?: number;
  selectedPlan?: OnboardingPlan;
  /** grandfathered users dismissed the "there is a wizard now" notice */
  dismissedUpgradePrompt?: boolean;
  /** set once by the v1 -> v2 migration: this profile was configured by hand
   * before the wizard existed, so the main window offers the upgrade notice.
   * A fresh profile that completed the wizard never carries it. */
  migratedFromV1?: boolean;
}

/** renderer -> main partial onboarding update (never touches `completed`) */
export interface OnboardingProgressPatch {
  lastStep?: number;
  selectedPlan?: OnboardingPlan;
  dismissedUpgradePrompt?: boolean;
}

/** wizard -> main: finish onboarding and hand over to the main window */
export interface OnboardingCompletePayload {
  selectedPlan?: OnboardingPlan;
}

/** identity of the running build (shown in the wizard footer / about) */
export interface AppInfo {
  version: string;
  platform: NodeJS.Platform;
  /** false in `npm run dev`, true inside an installed build */
  packaged: boolean;
}

export interface SettingsFile {
  version: 2;
  /** first-run wizard state; added in v2 (migrated files are grandfathered) */
  onboarding: OnboardingState;
  llm: {
    baseUrl: string;
    model: string;
    /** reply language for AI answers (R: 模式选择); default chinese */
    answerLang: AnswerLang;
    /** answer with the vision/multimodal provider instead of the text model */
    answerWithVision?: boolean;
    /**
     * 推理档位（DeepSeek 等支持 `reasoning_effort` 的模型）。
     *
     * ⚠️ 实时提词必须设 `'none'`：实测 `deepseek-flash` 默认 effort=high 时
     * 会把 max_tokens 全烧在推理链上（正文 0 字、首字 1.7s），
     * 传 `'none'` 后首字 453ms。缺省不传该字段 → 用服务端默认。
     */
    reasoningEffort?: ReasoningEffort;
    /** encrypted-at-rest (safeStorage, base64); never exposed raw to renderer */
    apiKeyEnc?: string;
    /** catalog provider behind baseUrl+model; 'custom' when unmatched */
    providerId?: ProviderId;
    /** last <=4 characters of the saved key, computed main-side at save time */
    apiKeyHint?: string;
    verification?: ProviderVerification;
  };
  vision: {
    baseUrl?: string;
    model?: string;
    apiKeyEnc?: string;
    /** proxy for blocked providers (e.g. Gemini): '127.0.0.1:7897'; empty = direct */
    proxyUrl?: string;
    providerId?: ProviderId;
    apiKeyHint?: string;
    verification?: ProviderVerification;
  };
  asr: {
    language: AsrLanguage;
    /** override models dir; default %APPDATA%/MeetingCopilot/models */
    modelsDir?: string;
    /** catalog provider behind the ACTIVE cloud slot; absent for local backends */
    providerId?: ProviderId;
    /** 'local-realtime' = auto-spawned local sidecar (FunASR or MOSS);
     * 'local' = whisper turbo on-device; 'cloud' = OpenAI-compatible ASR API;
     * 'cloud-realtime' = remote WebSocket streaming (e.g. Aliyun fun-asr-realtime) */
    backend?: 'local' | 'cloud' | 'cloud-realtime' | 'local-realtime';
    /** cloud ASR provider (used when backend === 'cloud') */
    cloud?: {
      baseUrl?: string;
      model?: string;
      apiKeyEnc?: string;
      apiKeyHint?: string;
      verification?: ProviderVerification;
    };
    /** streaming cloud ASR provider (used when backend === 'cloud-realtime');
     * separate slot so switching backends never clobbers the other's config */
    realtime?: {
      /** wss:// endpoint, e.g. wss://{ws}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference */
      baseUrl?: string;
      model?: string;
      apiKeyEnc?: string;
      apiKeyHint?: string;
      verification?: ProviderVerification;
    };
    /** local sidecar (backend === 'local-realtime'); fixed localhost endpoint,
     * model is FunASR Nano, paraformer streaming, or experimental MOSS */
    localRealtime?: {
      model?: string;
    };
  };
  ui: {
    stealth: boolean;
    hotkeyToggle: string;
    /** global hotkey for region-screenshot Q&A */
    hotkeyShot: string;
    opacity: number;
    /** answer-body font size (small=13px / medium=16px / large=19px) */
    fontScale: FontScale;
    theme: ThemeMode;
    /** UI display language; absent = follow OS locale (zh → zh, else en) */
    lang?: UiLang;
    /** start MeetingCopilot with the OS session; DEFAULT OFF, user opt-in only */
    autoLaunch?: boolean;
    /** the "still running in the tray" balloon was shown once; never repeated */
    trayNoticeShown?: boolean;
  };
  audio: {
    /** input used for the other-party channel on platforms without loopback */
    themDeviceId?: string;
    /** also capture the microphone (dual-channel transcription: 对方 + 我) */
    micEnabled: boolean;
    /** chosen mic device id ('' / undefined = system default) */
    micDeviceId?: string;
  };
  /** RAG（MeetingCopilot 2.0）：接地气检索增强。缺省视为按 DEFAULT_RAG_CONFIG 启用 */
  rag?: PersistedRagSettings;
}

/** 落盘的 RAG 配置（不含任何密钥明文）。 */
export interface PersistedRagSettings {
  enabled: boolean;
  topK: number;
  minScore: number;
  neighborExpand: boolean;
  requireCitations: boolean;
  embedding: EmbeddingSettings & {
    /**
     * embedding API Key 的密文（与 llm/vision/asr 槽位同一套 cipher）。
     * 必须显式声明：否则主进程只能用交集类型 + cast 落地，
     * 而 cast 会掩盖协议缺口（`apiKeySet` 的来源在类型上就不可查）。
     */
    apiKeyEnc?: string;
    /** key 末 4 位，仅用于界面回显，绝不回传明文 */
    apiKeyHint?: string;
  };
}

/** What the renderer is allowed to see (no secrets). */
export interface PublicSettings {
  version: 2;
  /**
   * true when the OS credential store is unavailable and API keys can only be
   * obfuscated (the plainCipher fallback). NOT persisted — it describes the
   * running machine, and both `settings:get` and `settings:set` report it so
   * the UI can warn BEFORE a key is sent for saving.
   */
  weakCrypto: boolean;
  /** first-run wizard state — public so the UI can show the upgrade notice */
  onboarding: OnboardingState;
  llm: {
    baseUrl: string;
    model: string;
    answerLang: AnswerLang;
    answerWithVision: boolean;
    /** 当前推理档位；缺省 = 不发送 reasoning_effort，由服务端决定 */
    reasoningEffort?: ReasoningEffort;
    apiKeySet: boolean;
    providerId?: ProviderId;
    /** last <=4 characters of the saved key — never the key itself */
    apiKeyHint?: string;
    verification?: ProviderVerification;
  };
  vision: {
    baseUrl?: string;
    model?: string;
    proxyUrl?: string;
    apiKeySet: boolean;
    providerId?: ProviderId;
    apiKeyHint?: string;
    verification?: ProviderVerification;
  };
  /** personal knowledge base (resume/notes) loaded from an .md file */
  knowledge: { chars: number };
  /** RAG 状态与配置（MeetingCopilot 2.0） */
  rag: {
    /** 是否启用检索增强；关闭时退回全量直塞的老路径 */
    enabled: boolean;
    /** 检索返回的证据条数上限 */
    topK: number;
    /**
     * 低置信阈值（原始余弦）。top-1 低于此值时**不调用 LLM**，
     * 直接返回固定兜底话术。需按自己的语料校准。
     */
    minScore: number;
    /** 邻接扩展：命中片段自动带上同源的相邻片段，保住被切散的上下文 */
    neighborExpand: boolean;
    /** 证据前是否强制要求编号引用 */
    requireCitations: boolean;
    embedding: EmbeddingSettings & { apiKeySet: boolean; apiKeyHint?: string };
  };
  asr: {
    language: AsrLanguage;
    modelsDir?: string;
    backend: 'local' | 'cloud' | 'cloud-realtime' | 'local-realtime';
    providerId?: ProviderId;
    cloud: {
      baseUrl?: string;
      model?: string;
      apiKeySet: boolean;
      apiKeyHint?: string;
      verification?: ProviderVerification;
    };
    realtime: {
      baseUrl?: string;
      model?: string;
      apiKeySet: boolean;
      apiKeyHint?: string;
      verification?: ProviderVerification;
    };
    localRealtime: { model?: string };
  };
  ui: {
    stealth: boolean;
    hotkeyToggle: string;
    hotkeyShot: string;
    opacity: number;
    fontScale: FontScale;
    theme: ThemeMode;
    lang: UiLang;
    autoLaunch: boolean;
    trayNoticeShown: boolean;
  };
  audio: { themDeviceId?: string; micEnabled: boolean; micDeviceId?: string };
}

/**
 * Renderer -> main settings update. Plaintext apiKey in transit only.
 * `apiKeyHint` is deliberately absent: it is derived main-side when a key is
 * saved, so the renderer can never desync (or spoof) it.
 */
export interface SettingsPatch {
  llm?: {
    baseUrl?: string;
    model?: string;
    answerLang?: AnswerLang;
    answerWithVision?: boolean;
    /** 推理档位；传 'none' 关掉思考模式（实时提词推荐） */
    reasoningEffort?: ReasoningEffort;
    apiKey?: string;
    providerId?: ProviderId;
    verification?: ProviderVerification;
  };
  vision?: {
    baseUrl?: string;
    model?: string;
    proxyUrl?: string;
    apiKey?: string;
    providerId?: ProviderId;
    verification?: ProviderVerification;
  };
  asr?: {
    language?: AsrLanguage;
    backend?: 'local' | 'cloud' | 'cloud-realtime' | 'local-realtime';
    providerId?: ProviderId;
    cloud?: {
      baseUrl?: string;
      model?: string;
      apiKey?: string;
      verification?: ProviderVerification;
    };
    realtime?: {
      baseUrl?: string;
      model?: string;
      apiKey?: string;
      verification?: ProviderVerification;
    };
    localRealtime?: { model?: string };
  };
  ui?: {
    stealth?: boolean;
    hotkeyToggle?: string;
    hotkeyShot?: string;
    opacity?: number;
    fontScale?: FontScale;
    theme?: ThemeMode;
    lang?: UiLang;
    autoLaunch?: boolean;
    trayNoticeShown?: boolean;
  };
  audio?: { themDeviceId?: string; micEnabled?: boolean; micDeviceId?: string };
  /** RAG 配置与 embedding 端点（apiKey 仅在此处一次性传入，落盘即加密） */
  rag?: {
    enabled?: boolean;
    topK?: number;
    minScore?: number;
    neighborExpand?: boolean;
    requireCitations?: boolean;
    /**
     * 必须是 Partial —— 界面经常只改其中一项（例如只换 API Key、只改模型，
     * 或点预设时只填 baseUrl+model）。若这里要求 EmbeddingSettings 全字段，
     * `setSettings({rag:{embedding:{apiKey}}})` 会被 tsc 挡死。
     * 主进程按 partial 合并到已有配置上。
     */
    embedding?: Partial<EmbeddingSettings> & { apiKey?: string };
  };
}

// ---------- ASR events (main -> renderer) ----------

export interface SegmentTimings {
  /** Date.now() of the first speech sample of the segment */
  speechStartTs: number;
  /** Date.now() of the last audio sample of the segment (speech end) */
  speechEndTs: number;
  /** when VAD closed the segment (speechEndTs + hangover) */
  vadCloseTs: number;
  inferStartTs: number;
  inferEndTs: number;
}

export interface AsrSegmentEvent {
  kind: 'segment';
  id: number;
  text: string;
  lang?: string;
  speaker: Speaker;
  audioMs: number;
  timings: SegmentTimings;
}

export interface AsrReadyEvent {
  kind: 'ready';
  loadMs: number;
  warmMs: number;
  ep: string;
  gpuSuspect: boolean; // true when warm timing suggests CPU fallback
}

export interface AsrStatusEvent {
  kind: 'status';
  state: 'loading' | 'listening' | 'speech' | 'transcribing' | 'stopped';
  queuedSegments: number;
}

export interface AsrErrorEvent {
  kind: 'error';
  message: string;
  fatal: boolean;
}

/** live streaming partial (transient; replaced by the final segment) */
export interface AsrPartialEvent {
  kind: 'partial';
  speaker: Speaker;
  text: string;
}

export type AsrEvent =
  | AsrSegmentEvent
  | AsrPartialEvent
  | AsrReadyEvent
  | AsrStatusEvent
  | AsrErrorEvent;

// ---------- Sessions (multi-conversation, persisted) ----------

export interface StoredTurn {
  id: string;
  kind: 'segment' | 'continuous' | 'free' | 'translate' | 'vision';
  label: string;
  text: string;
  status: 'streaming' | 'done' | 'error';
  error?: string;
}

export interface StoredSession {
  id: string;
  name: string;
  createdAt: number;
  turns: StoredTurn[];
  /** per-session transcript (isolated per meeting, same as the conversation) */
  segments?: import('./transcript').TranscriptSegment[];
  /** true once named (auto from first question, or manually renamed) */
  titled?: boolean;
  /**
   * 每槽多文档资料（三槽：简历 / 项目 / JD）。
   *
   * 取代原来的单份 `resumeText` / `jdText`：真实面试要同时带上简历、多个项目
   * 讲稿与目标岗位 JD，一份一份换太麻烦。旧的单份字段仍然保留在类型里以便
   * 读取老会话文件，加载时会迁移进这里（见 src/lib/sessionMaterial.ts）。
   */
  docs?: Partial<Record<KbSlot, KbDoc[]>>;
  /** legacy single-slot KB (pre dual-slot); migrated to the resume slot on load */
  kbName?: string;
  kbText?: string;
  /** legacy dual-slot session material: one resume + one JD (kept for old files) */
  resumeName?: string;
  resumeText?: string;
  jdName?: string;
  jdText?: string;
  /** rolling interview memo (P1): ≤800-char structured summary, async-updated */
  memo?: string;
}

export interface SessionsFile {
  sessions: StoredSession[];
  currentId: string | null;
}

/** 一次多选导入的结果：成功解析的文档 + 逐文件的失败原因（界面要能说清哪份没进来） */
export interface KbPickResult {
  docs: KbDoc[];
  /** 解析失败/无文本层的文件；scanned PDF 会落到这里 */
  errors: { name: string; reason: string }[];
}

// ---------- LLM (R4): renderer <-> main ----------

export interface LlmAskPayload {
  requestId: string;
  mode: 'segment' | 'continuous' | 'free' | 'translate';
  /** the sentence to answer (segment) or text to translate (translate) */
  question?: string;
  /** free-form question (mode === 'free') */
  freeQuestion?: string;
  /** recent transcript lines, oldest first */
  recentTranscript: string[];
  /** reply language for segment/continuous/free (translate is always zh) */
  answerLang?: AnswerLang;
  /** prior Q&A turns for session coherence (oldest first) */
  history?: { role: 'user' | 'assistant'; content: string }[];
  /** legacy single-slot KB (kept for compat; treated as resume material) */
  background?: string;
  /** dual-slot session material (P0-2) */
  resume?: string;
  jd?: string;
  /** rolling interview memo (P1) */
  memo?: string;
  /**
   * RAG 开关（单次请求级）。true 时主进程会用本轮的 `question` 检索知识库，
   * 把命中片段作为带编号的证据注入，并按结果决定是否走兜底话术。
   * 缺省 = false，保证老路径逐字节不变（前缀缓存不受影响）。
   */
  rag?: boolean;
  /**
   * 检索材料槽位：默认用 resume 槽（项目资料）。
   * 'jd' 时以岗位 JD 为主槽检索，便于"往 JD 上贴"的回答。
   */
  ragSlot?: KbSlot;
}

export type LlmEvent =
  | { requestId: string; kind: 'delta'; text: string }
  | { requestId: string; kind: 'done'; text: string }
  | { requestId: string; kind: 'error'; message: string }
  /** RAG：本次检索结果，先于答案推给界面，供"引用溯源"面板显示。
   *  摘要片段截断到 200 字，仅用于展示；真正喂给模型的全文在主进程内部。 */
  | { requestId: string; kind: 'retrieval'; retrieval: RagRetrieval; usedFallback: boolean }
  /**
   * RAG：答案生成完毕后的**引用覆盖率审计**（确定性规则，不额外调模型）。
   * 单独一条事件是因为必须等回答完整才能核对：
   * 界面据此提示"哪些事实句没有出处""引用了不存在的编号"。
   */
  | { requestId: string; kind: 'citation'; audit: CitationAudit };

// ---------- System tray (main -> renderer) ----------

/**
 * A tray menu entry the MAIN process cannot service on its own: capture,
 * sessions and every panel live in the renderer. Main always makes the window
 * visible first, so the renderer may assume it is on screen.
 */
export interface TrayCommandPayload {
  command: TrayRendererCommand;
}

// ---------- IPC channel names ----------

export const IPC = {
  /** renderer -> main, fire and forget: (ArrayBuffer pcmF32, captureTs, Speaker) */
  capturePcm: 'capture:pcm',
  /** invoke: (text) => string — cheap one-shot translation to Chinese (inline, off-session) */
  translateText: 'llm:translate',
  /** renderer -> main: capture lifecycle */
  captureStarted: 'capture:started',
  captureStopped: 'capture:stopped',
  /** main -> renderer: AsrEvent */
  asrEvent: 'asr:event',
  /** invoke: () => {ready, status} — pull the last ready/status AsrEvents.
   * Cloud engines are ready in ms, BEFORE React subscribes; push-replay on
   * did-finish-load still races the subscription, so the renderer pulls. */
  asrReplay: 'asr:replay',
  /** invoke: () => PublicSettings */
  settingsGet: 'settings:get',
  /** invoke: (SettingsPatch) => PublicSettings */
  settingsSet: 'settings:set',
  /** invoke: () => {chars:number} — import a .md into the GLOBAL default KB (opens dialog) */
  knowledgeImport: 'knowledge:import',
  /** invoke: () => {chars:number} — clear the global default KB */
  knowledgeClear: 'knowledge:clear',
  /**
   * invoke: (KbSlot) => KbPickResult — 为**当前会话**的某个槽位选文档。
   * 支持**多选**：一次可以挑多份讲稿/手册，逐份确定性解析（不经过 LLM）。
   */
  knowledgePick: 'knowledge:pick',
  /** invoke: () => SessionsFile — load persisted sessions */
  sessionsLoad: 'sessions:load',
  /** send: (SessionsFile) — persist sessions (debounced by renderer) */
  sessionsSave: 'sessions:save',
  /** invoke: () => string|null — full-screen capture, drag a region (stealth overlay), returns cropped dataURL */
  regionPick: 'region:pick',
  /** invoke (overlay→main): () => string|null — the captured full-screen image to draw */
  regionImage: 'region:image',
  /** send (overlay→main): (rect) — chosen region */
  regionRect: 'region:rect',
  /** send (overlay→main): () — cancel selection */
  regionCancel: 'region:cancel',
  /** invoke: () => string|null — PNG dataURL of the clipboard image (Win+Shift+S
   * snip), read on an explicit Ctrl+V paste; never polled */
  clipboardReadImage: 'app:clipboard-read-image',
  /** invoke: (boolean) => boolean — toggles content protection live */
  stealthSet: 'stealth:set',
  /** send: hide window */
  winHide: 'win:hide',
  /** send: quit app (clean) */
  appQuit: 'app:quit',
  /** main -> renderer: request auto start capture (dev/E2E) */
  autoStart: 'capture:auto-start',
  /** main -> renderer: the screenshot hotkey was pressed */
  shotHotkey: 'shot:hotkey',
  /** renderer -> main: start a streaming answer (LlmAskPayload) */
  llmAsk: 'llm:ask',
  /** renderer -> main: screenshot + vision question ({requestId, question}); answer streams on llmEvent */
  shotAsk: 'shot:ask',
  /** renderer -> main: cancel a running request (requestId) */
  llmCancel: 'llm:cancel',
  /** main -> renderer: LlmEvent stream */
  llmEvent: 'llm:event',
  /** send: ({resume?, jd?}) — warm the DeepSeek KV prefix cache (P1-6):
   * one max_tokens=1 request whose system prompt is byte-identical to real
   * answer requests, so the first real question prefills from cache */
  llmPrewarm: 'llm:prewarm',
  /** invoke: ({memo, question, answer}) => string — async rolling interview
   * memo update (P1-5); cheap off-critical-path deepseek-chat call, '' = keep old */
  memoUpdate: 'llm:memo',
  /** invoke: () => OnboardingState — first-run wizard state */
  onboardingGet: 'onboarding:get',
  /** invoke: (OnboardingProgressPatch) => OnboardingState — 保存并稍后继续 /
   * dismissing the upgrade notice; never flips `completed` */
  onboardingSaveProgress: 'onboarding:save-progress',
  /** invoke: (OnboardingCompletePayload) => OnboardingState — wizard finished:
   * persist completion, close the setup window and hand over to the main app */
  onboardingComplete: 'onboarding:complete',
  /** invoke: () => boolean — main window asks for the wizard again ("重新运行
   * 配置向导" / the upgrade notice). Re-run mode keeps the main window alive
   * and never quits the app when the wizard is closed. */
  onboardingRerun: 'onboarding:rerun',
  /** invoke: (url) => boolean — open an allowlisted https URL in the OS
   * browser; the renderer can never navigate or window.open by itself */
  externalOpen: 'app:open-external',
  /** invoke: () => string — read the clipboard, ONLY from an explicit
   * paste-button click (never polled) */
  clipboardReadText: 'app:clipboard-read',
  /** invoke: () => AppInfo */
  appGetInfo: 'app:get-info',
  /** invoke: (ProviderTestRequest) => ProviderTestResult — run one real
   * connection test against a provider. Explicit user action only: never on
   * startup, never polled, one minimal billable request per call. Main also
   * records the verdict into the slot's `verification` (a dedicated store
   * write that deliberately does NOT go through settings:set, which would
   * restart the ASR engine). */
  providerTest: 'provider:test',
  /** invoke: () => string — plaintext, locally built support report. Contains
   * no keys, no transcripts and no knowledge-base text (electron/diagnostics.ts) */
  diagnosticsGet: 'diagnostics:get',
  /** invoke: () => boolean — reveal the userData folder in Explorer/Finder */
  logsOpenFolder: 'logs:open-folder',
  /** main -> renderer: TrayCommandPayload — a tray menu entry that only the
   * renderer can service (start/stop capture, new session, open a panel).
   * Main shows the window before sending, so the UI is always visible. */
  trayCommand: 'tray:command',

  /** invoke: () => RagIndexInfo — 知识库索引总览 */
  ragIndexGet: 'rag:index-get',
  /** invoke: ({slot, source, text}) => RagIndexInfo — 用已解析的文本重建该槽索引 */
  ragIndexRebuild: 'rag:index-rebuild',
  /** invoke: () => RagIndexInfo — 清空全部索引（不动原始资料） */
  ragIndexClear: 'rag:index-clear',
  /** invoke: ({query, topK}) => RagRetrieval — 检索自测：看命中什么、分数多少 */
  ragSearchTest: 'rag:search-test',
  /** invoke: ({apiKey?, baseUrl, model}) => {ok, code, messageZh, dim?, latencyMs} — embedding 连通性测试 */
  ragEmbeddingTest: 'rag:embedding-test',
  /** main -> renderer: RagIndexProgress — 索引进度（流式） */
  ragProgress: 'rag:progress',

  /**
   * invoke: ({deep?}) => ReadinessReport
   * 面试前自检。`deep: true` 时会额外发起一次 embedding 连通性请求
   * （快速自检不发任何网络请求）。
   */
  readinessCheck: 'readiness:check',
} as const;
