/**
 * Embedding 预设与 RAG 默认值 —— **单一事实源**。
 *
 * 放在 `shared/` 是因为主进程与渲染进程都要用：
 *   * 主进程 `electron/rag/embedder.ts` 用它做端点/模型默认值；
 *   * 渲染进程设置面板用它渲染预设下拉与回显。
 * 之前渲染侧复制了一份副本，两边一旦漂移，界面显示的预设就会和后端
 * 实际请求的端点不一致（用户看到"阿里云百炼"却打到别的地址）——
 * 这类 bug 极难排查，所以必须是同一份常量。
 *
 * 本文件只允许包含**纯数据与纯类型**：不能 import 任何 electron/fs/fetch
 * 相关模块，否则会污染渲染进程的打包结果。
 */

export interface EmbeddingProviderPreset {
  id: string;
  labelZh: string;
  baseUrl: string;
  model: string;
  dim: number;
  /** 控制台取 key 的入口，显示在设置面板 */
  keyUrl: string;
  noteZh: string;
}

/**
 * 默认选中百炼 text-embedding-v3：用户已有 DashScope key（正在用于
 * fun-asr-realtime 实时转写），**同一个 key 直接复用**，不需要再注册账号
 * —— 这是选用它而不是本地 Ollama 的主要原因。
 */
export const EMBEDDING_PRESETS: EmbeddingProviderPreset[] = [
  {
    id: 'dashscope',
    labelZh: '阿里云百炼 text-embedding-v3（推荐，与转写同 key）',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'text-embedding-v3',
    dim: 1024,
    keyUrl: 'https://bailian.console.aliyun.com/?apiKey=1',
    noteZh: '中文效果强，1024 维。与你正在用的实时转写共用同一个 DashScope API Key。',
  },
  {
    id: 'siliconflow',
    labelZh: '硅基流动 BAAI/bge-m3',
    baseUrl: 'https://api.siliconflow.cn/v1',
    model: 'BAAI/bge-m3',
    dim: 1024,
    keyUrl: 'https://cloud.siliconflow.cn/account/ak',
    noteZh: '中文优秀，有免费额度；bge-m3 支持长文本与多语言。',
  },
  {
    id: 'openai',
    labelZh: 'OpenAI text-embedding-3-small',
    baseUrl: 'https://api.openai.com/v1',
    model: 'text-embedding-3-small',
    dim: 1536,
    keyUrl: 'https://platform.openai.com/api-keys',
    noteZh: '国内访问需自备代理；维度 1536。',
  },
  {
    id: 'custom',
    labelZh: '自定义（任意 OpenAI 兼容端点）',
    baseUrl: '',
    model: '',
    dim: 1024,
    keyUrl: '',
    noteZh: '填写 baseUrl（到 /v1 为止）与模型名，例如本地 vLLM、LM Studio、其他云厂商。',
  },
];

export const DEFAULT_EMBEDDING_PRESET = EMBEDDING_PRESETS[0];

/**
 * 低置信阈值默认值（原始余弦）。
 * ⚠️ 这是**初始估值，没有实测依据**：中文同域文本的余弦基线常在 0.3–0.6，
 * 因此该值必须用你自己的语料校准（见 `test/rag-e2e-live.spec.ts`）。
 */
export const DEFAULT_MIN_SCORE = 0.45;
/** 检索返回的证据条数上限 */
export const DEFAULT_TOP_K = 5;

/** baseUrl+model → 预设（用于下拉回显；匹配不到视为「自定义」） */
export function findEmbeddingPreset(
  baseUrl: string,
  model: string,
): EmbeddingProviderPreset | undefined {
  const b = baseUrl.trim().replace(/\/+$/, '');
  const m = model.trim();
  return EMBEDDING_PRESETS.find(
    (p) => p.baseUrl !== '' && p.baseUrl.replace(/\/+$/, '') === b && p.model === m,
  );
}
