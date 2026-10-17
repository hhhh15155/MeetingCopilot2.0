/**
 * 渲染进程侧的 RAG 默认值与兜底。
 *
 * 预设表（EMBEDDING_PRESETS / DEFAULT_MIN_SCORE / DEFAULT_TOP_K / findEmbeddingPreset）
 * 的**单一事实源在 `shared/ragDefaults.ts`**，主进程与渲染进程共用同一份，
 * 这里只做转发——否则界面显示的预设可能与后端实际请求的端点漂移，
 * 这类 bug 表现为"我明明选了阿里云却打到别的地址"，极难排查。
 */
import type { PublicSettings } from '../../shared/protocol';
import {
  DEFAULT_EMBEDDING_PRESET,
  DEFAULT_MIN_SCORE,
  DEFAULT_TOP_K,
} from '../../shared/ragDefaults';

export {
  EMBEDDING_PRESETS,
  DEFAULT_EMBEDDING_PRESET,
  DEFAULT_MIN_SCORE,
  DEFAULT_TOP_K,
  findEmbeddingPreset,
} from '../../shared/ragDefaults';
export type { EmbeddingProviderPreset } from '../../shared/ragDefaults';

/**
 * `PublicSettings.rag` 的兜底值：settings 还没加载完（首帧）时界面仍然可用。
 * 取值与主进程 `electron/rag/index.ts` 的 `DEFAULT_RAG_CONFIG` 保持一致。
 */
export const DEFAULT_RAG_SETTINGS: PublicSettings['rag'] = {
  enabled: true,
  topK: DEFAULT_TOP_K,
  minScore: DEFAULT_MIN_SCORE,
  neighborExpand: true,
  requireCitations: true,
  embedding: {
    providerId: DEFAULT_EMBEDDING_PRESET.id,
    baseUrl: DEFAULT_EMBEDDING_PRESET.baseUrl,
    model: DEFAULT_EMBEDDING_PRESET.model,
    dim: DEFAULT_EMBEDDING_PRESET.dim,
    apiKeySet: false,
  },
};

/** 主进程回传缺字段时补齐，避免面板读到 undefined 崩掉 */
export function ragSettingsOf(settings: PublicSettings | null | undefined): PublicSettings['rag'] {
  const raw = settings?.rag;
  if (!raw) return DEFAULT_RAG_SETTINGS;
  return {
    ...DEFAULT_RAG_SETTINGS,
    ...raw,
    embedding: { ...DEFAULT_RAG_SETTINGS.embedding, ...raw.embedding },
  };
}
