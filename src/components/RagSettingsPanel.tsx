import { useCallback, useEffect, useState } from 'react';
import type {
  KbDoc,
  KbSlot,
  PublicSettings,
  RagIndexInfo,
  RagIndexProgress,
  RagRetrieval,
  SettingsPatch,
} from '../../shared/protocol';
import { KB_SLOTS } from '../../shared/protocol';
import { sanitizeApiKeyInput } from '../../shared/keyInput';
import { useT } from '../i18n';
import {
  DEFAULT_MIN_SCORE,
  DEFAULT_TOP_K,
  EMBEDDING_PRESETS,
  findEmbeddingPreset,
  ragSettingsOf,
} from '../rag/embeddingPresets';
import { EvidenceRow } from './RagEvidenceView';

/**
 * 当前会话已导入的资料（重建索引用）。
 *
 * 三槽多文档：`docs` 每槽一个数组。界面与索引都只看这里，
 * 不直接读会话上的 legacy 单份字段（那些由 shared/sessionMaterial 迁移）。
 */
export interface RagMaterial {
  docs: Record<KbSlot, KbDoc[]>;
}

/**
 * 索引内容发生变化（重建完成 / 清空）时派发。
 *
 * 为什么需要它：`ragIndexClear` 不会产生索引进度事件，App 无法得知索引已空。
 * 而 App 必须知道索引是否就绪——否则会继续发 `rag: true`，主进程检索不到
 * 就直接返回兜底话术，表现为"清空索引后每一题都答不上来"。
 */
export const RAG_INDEX_CHANGED = 'mc:rag-index-changed';

const notifyIndexChanged = () => window.dispatchEvent(new Event(RAG_INDEX_CHANGED));

const toInt = (s: string, fallback: number): number => {
  const n = Number.parseInt(s, 10);
  return Number.isFinite(n) ? n : fallback;
};

const toNum = (s: string, fallback: number): number => {
  const n = Number(s);
  return Number.isFinite(n) ? n : fallback;
};

const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n));

/** embedding 连通性测试的界面状态 */
interface EmbeddingTestState {
  testing: boolean;
  ok?: boolean;
  messageZh?: string;
  dim?: number;
  latencyMs?: number;
}

/**
 * 「检索增强 / RAG」设置面板（MeetingCopilot 2.0）。
 *
 * 组织方式：总开关 → Embedding 配置（含真实连通性测试）→ 索引管理
 * （总览 / 重建 / 进度 / 清空）→ 检索自测（显示**原始余弦**）→
 * 低置信阈值与 topK。所有文案走 i18n，样式与设置面板一致。
 */
export function RagSettingsPanel({
  settings,
  material,
  ragSlot,
  onRagSlotChange,
  onSettingsChange,
}: {
  settings: PublicSettings;
  material: RagMaterial;
  /** 提问时优先检索的槽位（App 持有，不落盘） */
  ragSlot: KbSlot;
  onRagSlotChange: (slot: KbSlot) => void;
  /** 保存成功后回传最新 settings（面板保持打开） */
  onSettingsChange: (s: PublicSettings) => void;
}) {
  const t = useT();
  const rag = ragSettingsOf(settings);

  // ---- 待保存草稿（面板是草稿，点「保存设置」才提交） ----
  const [enabled, setEnabled] = useState(rag.enabled);
  const [topK, setTopK] = useState(String(rag.topK));
  const [minScore, setMinScore] = useState(String(rag.minScore));
  const [neighborExpand, setNeighborExpand] = useState(rag.neighborExpand);
  const [requireCitations, setRequireCitations] = useState(rag.requireCitations);
  const [presetId, setPresetId] = useState(
    () => findEmbeddingPreset(rag.embedding.baseUrl, rag.embedding.model)?.id ?? 'custom',
  );
  const [baseUrl, setBaseUrl] = useState(rag.embedding.baseUrl);
  const [model, setModel] = useState(rag.embedding.model);
  const [dim, setDim] = useState(rag.embedding.dim ? String(rag.embedding.dim) : '');
  const [keyValue, setKeyValue] = useState('');
  const [keyEditing, setKeyEditing] = useState(false);
  const [keyPendingDelete, setKeyPendingDelete] = useState(false);
  const [keyConfirming, setKeyConfirming] = useState(false);

  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [confirmWeak, setConfirmWeak] = useState(false);

  // ---- 索引状态 ----
  const [info, setInfo] = useState<RagIndexInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [progress, setProgress] = useState<RagIndexProgress | null>(null);
  const [action, setAction] = useState<'rebuild' | 'clear' | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);

  // ---- 测试 ----
  const [embedTest, setEmbedTest] = useState<EmbeddingTestState>({ testing: false });
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<RagRetrieval | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const dimNum = dim.trim() === '' ? 0 : clamp(toInt(dim, 0), 0, 8192);
  const topKNum = clamp(toInt(topK, DEFAULT_TOP_K), 1, 20);
  const minScoreNum = clamp(toNum(minScore, DEFAULT_MIN_SCORE), 0, 1);
  const draftKey = sanitizeApiKeyInput(keyValue).value;

  const refreshInfo = useCallback(async () => {
    try {
      setInfo(await window.mc.ragIndexGet());
      setInfoError(null);
    } catch (e) {
      setInfoError(t.rag.loadFailed((e as Error).message));
    }
  }, [t]);

  useEffect(() => {
    void refreshInfo();
  }, [refreshInfo]);

  // 索引进度（主进程流式推送）：完成/出错时刷新总览
  useEffect(() => {
    return window.mc.onRagProgress((p) => {
      setProgress(p);
      if (p.status === 'ready') {
        setAction(null);
        void refreshInfo();
      } else if (p.status === 'error') {
        setAction(null);
      }
    });
  }, [refreshInfo]);

  // ---- 保存 ----
  const patchForSave = (): SettingsPatch['rag'] => ({
    enabled,
    topK: topKNum,
    minScore: minScoreNum,
    neighborExpand,
    requireCitations,
    embedding: {
      providerId: presetId,
      baseUrl: baseUrl.trim(),
      model: model.trim(),
      // 0 = 用服务端默认维度（protocol 约定），也用于把旧的 dim 清掉
      dim: dimNum,
      ...(draftKey || keyPendingDelete ? { apiKey: keyPendingDelete ? '' : draftKey } : {}),
    },
  });

  const save = async () => {
    setConfirmWeak(false);
    setSaving(true);
    setSaveError(null);
    try {
      const next = await window.mc.setSettings({ rag: patchForSave() });
      setKeyValue('');
      setKeyEditing(false);
      setKeyPendingDelete(false);
      setKeyConfirming(false);
      setSavedAt(Date.now());
      onSettingsChange(next);
      void refreshInfo();
    } catch (e) {
      setSaveError(t.rag.saveFailed((e as Error).message));
    } finally {
      setSaving(false);
    }
  };

  const requestSave = () => {
    // 与设置面板同一套弱加密预检：key 明文离开渲染进程前先问一次
    if (settings.weakCrypto && draftKey) {
      setConfirmWeak(true);
      return;
    }
    void save();
  };

  // ---- 重建 / 清空 ----
  /**
   * 收集要索引的文档：**三槽 × 每槽多份**，逐份建索引。
   *
   * 逐份而不是把整槽拼成一大段：RagStore 以 `(slot, source)` 为键，
   * 一份文档一个 source 才能各自显示进度、各自统计片段数，
   * 检索时也才能告诉你"这条依据来自哪份资料的哪一段"。
   */
  const rebuildTargets = (): { slot: KbSlot; source: string; text: string }[] => {
    const out: { slot: KbSlot; source: string; text: string }[] = [];
    for (const slot of KB_SLOTS) {
      for (const d of material.docs[slot] ?? []) {
        if (d.text.trim()) out.push({ slot, source: d.name, text: d.text });
      }
    }
    return out;
  };
  const targets = rebuildTargets();
  /** 槽位 → 中文名（三槽统一走这里，避免各处各写一份三元表达式） */
  const slotLabel = (slot: KbSlot): string =>
    slot === 'jd' ? t.rag.slotJd : slot === 'project' ? t.rag.slotProject : t.rag.slotResume;

  const rebuild = async () => {
    setAction('rebuild');
    setActionMsg(null);
    setActionError(null);
    setProgress(null);
    try {
      for (const target of targets) {
        // 逐份建索引：进度通过 onRagProgress 流式到达
        const r = await window.mc.ragIndexRebuild(target);
        if ('error' in r) {
          setActionError(t.rag.rebuildFailed(`${target.source}：${r.error}`));
          setAction(null);
          return;
        }
        setInfo(r);
      }
      setActionMsg(t.rag.rebuildDone(targets.length));
      notifyIndexChanged();
    } catch (e) {
      setActionError(t.rag.rebuildFailed((e as Error).message));
    } finally {
      setAction(null);
    }
  };

  const clearIndex = async () => {
    setAction('clear');
    setActionMsg(null);
    setActionError(null);
    setConfirmClear(false);
    try {
      setInfo(await window.mc.ragIndexClear());
      setProgress(null);
      setActionMsg(t.rag.clearDone);
      // 清空不发进度事件，必须显式通知 App：否则它会继续发 rag:true 并每问吃兜底
      notifyIndexChanged();
    } catch (e) {
      setActionError(t.rag.clearFailed((e as Error).message));
    } finally {
      setAction(null);
    }
  };

  // ---- 测试 ----
  const runEmbedTest = async () => {
    setEmbedTest({ testing: true });
    try {
      const r = await window.mc.ragEmbeddingTest({
        ...(draftKey ? { apiKey: draftKey } : {}),
        baseUrl: baseUrl.trim(),
        model: model.trim(),
        ...(dimNum > 0 ? { dim: dimNum } : {}),
      });
      setEmbedTest({ testing: false, ...r });
    } catch (e) {
      setEmbedTest({ testing: false, ok: false, messageZh: (e as Error).message });
    }
  };

  const runSearch = async () => {
    if (!query.trim()) {
      setSearchError(t.rag.searchNeedQuery);
      setSearchResult(null);
      return;
    }
    setSearching(true);
    setSearchError(null);
    setSearchResult(null);
    try {
      const r = await window.mc.ragSearchTest({ query: query.trim(), topK: topKNum });
      if ('error' in r) setSearchError(t.rag.searchFailed(r.error));
      else setSearchResult(r);
    } catch (e) {
      setSearchError(t.rag.searchFailed((e as Error).message));
    } finally {
      setSearching(false);
    }
  };

  // ---- 派生 ----
  const keyUrl =
    EMBEDDING_PRESETS.find((p) => p.id === presetId)?.keyUrl ?? EMBEDDING_PRESETS[0].keyUrl;
  const presetNote = EMBEDDING_PRESETS.find((p) => p.id === presetId)?.noteZh;
  const hasIndex = (info?.chunkCount ?? 0) > 0;
  const dirty =
    enabled !== rag.enabled ||
    topKNum !== rag.topK ||
    minScoreNum !== rag.minScore ||
    neighborExpand !== rag.neighborExpand ||
    requireCitations !== rag.requireCitations ||
    baseUrl.trim() !== rag.embedding.baseUrl ||
    model.trim() !== rag.embedding.model ||
    dimNum !== (rag.embedding.dim ?? 0) ||
    draftKey !== '' ||
    keyPendingDelete;
  const canTestEmbedding = !embedTest.testing && (draftKey !== '' || rag.embedding.apiKeySet);
  const progressPct =
    progress && progress.chunksTotal > 0
      ? Math.round((progress.chunksEmbedded / progress.chunksTotal) * 100)
      : 0;

  const selectBool = (value: boolean, onChange: (v: boolean) => void) => (
    <select value={value ? 'on' : 'off'} onChange={(e) => onChange(e.target.value === 'on')}>
      <option value="on">{t.rag.on}</option>
      <option value="off">{t.rag.off}</option>
    </select>
  );

  return (
    <>
      <div className="settings-section">{t.rag.title}</div>
      <div className="settings-hint">{t.rag.intro}</div>

      {/* ---------------- 总开关 ---------------- */}
      <div className="settings-section">{t.rag.enableSection}</div>
      <div className="settings-row">
        <label>{t.rag.enable}</label>
        <select value={enabled ? 'on' : 'off'} onChange={(e) => setEnabled(e.target.value === 'on')}>
          <option value="on">{t.rag.enableOn}</option>
          <option value="off">{t.rag.enableOff}</option>
        </select>
        <span className="settings-inline-hint">{t.rag.enableHint}</span>
      </div>
      {!enabled && <div className="settings-hint">{t.rag.offNotice}</div>}

      {/* ---------------- Embedding 配置 ---------------- */}
      <div className="settings-section">{t.rag.embeddingSection}</div>
      <div className="settings-hint">{t.rag.embeddingIntro}</div>
      <div className="settings-row">
        <label>{t.rag.preset}</label>
        <select
          value={presetId}
          onChange={(e) => {
            const p = EMBEDDING_PRESETS.find((x) => x.id === e.target.value);
            setPresetId(e.target.value);
            if (p && p.id !== 'custom') {
              setBaseUrl(p.baseUrl);
              setModel(p.model);
              setDim(String(p.dim));
            }
          }}
        >
          {EMBEDDING_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.labelZh}
            </option>
          ))}
        </select>
        {presetNote && <span className="settings-inline-hint">{presetNote}</span>}
      </div>
      <div className="settings-row">
        <label>{t.rag.baseUrl}</label>
        <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} spellCheck={false} />
      </div>
      <div className="settings-row">
        <label>{t.rag.model}</label>
        <input value={model} onChange={(e) => setModel(e.target.value)} spellCheck={false} />
      </div>
      <div className="settings-row">
        <label>{t.rag.dim}</label>
        <input
          value={dim}
          inputMode="numeric"
          spellCheck={false}
          placeholder={t.rag.dimPlaceholder}
          onChange={(e) => setDim(e.target.value)}
        />
      </div>
      <div className="settings-row">
        <label>{t.rag.apiKey}</label>
        {rag.embedding.apiKeySet && !keyEditing && !keyPendingDelete ? (
          <div className="key-status">
            <span className="tag tag-ok">
              {rag.embedding.apiKeyHint
                ? t.rag.keySaved(rag.embedding.apiKeyHint)
                : t.rag.keySet}
            </span>
            <button className="btn btn-sm" onClick={() => setKeyEditing(true)}>
              {t.rag.keyReplace}
            </button>
            {keyConfirming ? (
              <>
                <span className="settings-inline-hint">{t.rag.keyDeleteConfirm}</span>
                <button
                  className="btn btn-sm"
                  onClick={() => {
                    setKeyPendingDelete(true);
                    setKeyConfirming(false);
                  }}
                >
                  {t.rag.keyDeleteYes}
                </button>
                <button className="btn btn-sm" onClick={() => setKeyConfirming(false)}>
                  {t.rag.keyDeleteNo}
                </button>
              </>
            ) : (
              <button className="btn btn-sm" onClick={() => setKeyConfirming(true)}>
                {t.rag.keyDelete}
              </button>
            )}
          </div>
        ) : keyPendingDelete ? (
          <div className="key-status">
            <span className="tag tag-err">{t.rag.keyPendingDelete}</span>
            <button className="btn btn-sm" onClick={() => setKeyPendingDelete(false)}>
              {t.rag.keyUndoDelete}
            </button>
          </div>
        ) : (
          <>
            <input
              type="password"
              value={keyValue}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setKeyValue(e.target.value)}
              placeholder={
                rag.embedding.apiKeySet ? t.rag.keyKeepPlaceholder : t.rag.keyPlaceholder
              }
            />
            <span className="settings-inline-hint">
              {rag.embedding.apiKeySet ? t.rag.keySet : t.rag.keyMissing} · {t.rag.keyHint}
            </span>
          </>
        )}
      </div>
      <div className="settings-row">
        <div className="key-status">
          <button
            className="btn btn-sm"
            disabled={!canTestEmbedding}
            title={canTestEmbedding ? undefined : t.rag.testNeedKey}
            onClick={() => void runEmbedTest()}
          >
            {embedTest.testing ? t.rag.testing : t.rag.test}
          </button>
          {keyUrl && (
            <button className="btn btn-sm" onClick={() => void window.mc.openExternal(keyUrl)}>
              {t.rag.openKeyPage}
            </button>
          )}
        </div>
        <span className="settings-inline-hint">{t.rag.testNote}</span>
      </div>
      {(embedTest.testing || embedTest.ok !== undefined || embedTest.messageZh) && (
        <div className={embedTest.ok ? 'rag-result is-ok' : 'rag-result is-err'}>
          <span className="rag-result-tag">
            {embedTest.testing
              ? t.rag.testing
              : embedTest.ok
                ? t.rag.testOk
                : t.rag.testFail}
          </span>
          {embedTest.messageZh && <span className="rag-result-msg">{embedTest.messageZh}</span>}
          {embedTest.ok && embedTest.dim !== undefined && embedTest.latencyMs !== undefined && (
            <span className="rag-result-meta">
              {t.rag.testMeta(embedTest.dim, embedTest.latencyMs)}
            </span>
          )}
        </div>
      )}

      {/* ---------------- 索引管理 ---------------- */}
      <div className="settings-section">{t.rag.indexSection}</div>
      <div className="settings-hint">{t.rag.indexIntro}</div>
      {infoError && <div className="settings-warn">{infoError}</div>}
      {info && hasIndex ? (
        <>
          <div className="rag-index-grid">
            <span>{t.rag.indexChunks(info.chunkCount)}</span>
            <span>{t.rag.indexSources(info.sources.length)}</span>
            {info.model && <span>{t.rag.indexModel(info.model)}</span>}
            {info.dim ? <span>{t.rag.indexDim(info.dim)}</span> : null}
            <span>
              {info.indexedAt
                ? t.rag.indexAt(new Date(info.indexedAt).toLocaleString(t.locale))
                : t.rag.indexUnknown}
            </span>
          </div>
          {info.sources.length > 0 && (
            <ul className="rag-source-list">
              {info.sources.map((s) => (
                <li key={`${s.slot}-${s.name}`} className="rag-source-item">
                  {t.rag.sourceRow(slotLabel(s.slot), s.name, s.chunks)}
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <div className="settings-inline-hint">{t.rag.indexEmpty}</div>
      )}
      <div className="key-status">
        <button className="btn btn-sm" onClick={() => void refreshInfo()}>
          {t.rag.refresh}
        </button>
        <button
          className="btn btn-sm"
          disabled={action !== null || targets.length === 0}
          title={targets.length === 0 ? t.rag.rebuildNoMaterial : t.rag.rebuildHint}
          onClick={() => void rebuild()}
        >
          {action === 'rebuild' ? t.rag.rebuilding : t.rag.rebuild}
        </button>
        {confirmClear ? (
          <>
            <span className="settings-inline-hint">{t.rag.clearHint}</span>
            <button className="btn btn-sm" disabled={action !== null} onClick={() => void clearIndex()}>
              {t.rag.clearConfirm}
            </button>
            <button className="btn btn-sm" onClick={() => setConfirmClear(false)}>
              {t.rag.clearCancel}
            </button>
          </>
        ) : (
          <button
            className="btn btn-sm"
            disabled={action !== null || !hasIndex}
            onClick={() => setConfirmClear(true)}
          >
            {t.rag.clear}
          </button>
        )}
      </div>
      {targets.length === 0 && (
        <div className="settings-inline-hint">{t.rag.rebuildNoMaterial}</div>
      )}
      {draftKey !== '' && <div className="settings-warn">{t.rag.rebuildUnsavedKey}</div>}
      <div className="settings-inline-hint">{t.rag.rebuildHint}</div>

      {progress && (
        <div className="rag-progress">
          <div className="rag-progress-track">
            <div
              className={
                progress.status === 'error'
                  ? 'rag-progress-fill is-error'
                  : progress.status === 'ready'
                    ? 'rag-progress-fill is-done'
                    : 'rag-progress-fill'
              }
              style={{ width: `${progress.status === 'error' ? 100 : progressPct}%` }}
            />
          </div>
          <span className="rag-progress-label">
            {t.rag.progressStatus[progress.status]} ·{' '}
            {t.rag.progressCount(progress.chunksEmbedded, progress.chunksTotal)}
          </span>
          {progress.source && (
            <span className="rag-progress-source">{t.rag.progressSource(progress.source)}</span>
          )}
        </div>
      )}
      {progress?.status === 'error' && progress.error && (
        <div className="rag-result is-err">{t.rag.progressError(progress.error)}</div>
      )}
      {actionMsg && <div className="rag-result is-ok">{actionMsg}</div>}
      {actionError && <div className="rag-result is-err">{actionError}</div>}

      {/* ---------------- 检索自测 ---------------- */}
      <div className="settings-section">{t.rag.searchSection}</div>
      <div className="settings-hint">{t.rag.searchIntro}</div>
      <div className="settings-row">
        <label>{t.rag.search}</label>
        <div className="key-status">
          <input
            value={query}
            spellCheck={false}
            placeholder={t.rag.searchPlaceholder}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.nativeEvent.isComposing) void runSearch();
            }}
          />
          <button
            className="btn btn-sm"
            disabled={searching || !hasIndex}
            title={hasIndex ? undefined : t.rag.searchNeedIndex}
            onClick={() => void runSearch()}
          >
            {searching ? t.rag.searching : t.rag.search}
          </button>
        </div>
        {!hasIndex && <span className="settings-inline-hint">{t.rag.searchNeedIndex}</span>}
      </div>
      {searchError && <div className="rag-result is-err">{searchError}</div>}
      {searchResult && (
        <>
          <div className="rag-index-grid">
            <span>{t.rag.searchHits(searchResult.evidence.length)}</span>
            <span>{t.rag.searchTop(searchResult.topScore.toFixed(3))}</span>
            <span>{t.rag.searchLatency(searchResult.latencyMs)}</span>
          </div>
          {searchResult.evidence.length === 0 ? (
            <div className="settings-inline-hint">{t.rag.searchEmpty}</div>
          ) : (
            <ul className="rag-evidence-list">
              {searchResult.evidence.map((ev) => (
                <EvidenceRow
                  key={`${ev.ref}-${ev.chunkId}`}
                  ev={ev}
                  minScore={minScoreNum}
                  chars={120}
                  showBonus
                />
              ))}
            </ul>
          )}
          <div className="settings-inline-hint">
            {searchResult.topScore < minScoreNum
              ? t.rag.searchLowConfidence(minScoreNum.toFixed(2))
              : t.rag.searchHighConfidence(minScoreNum.toFixed(2))}
          </div>
        </>
      )}

      {/* ---------------- 阈值与条数 ---------------- */}
      <div className="settings-section">{t.rag.thresholdSection}</div>
      <div className="settings-row">
        <label>{t.rag.minScore}</label>
        <input
          value={minScore}
          inputMode="decimal"
          spellCheck={false}
          onChange={(e) => setMinScore(e.target.value)}
        />
        <span className="settings-inline-hint">{t.rag.minScoreHint}</span>
      </div>
      <div className="settings-row">
        <label>{t.rag.topK}</label>
        <input
          value={topK}
          inputMode="numeric"
          spellCheck={false}
          onChange={(e) => setTopK(e.target.value)}
        />
        <span className="settings-inline-hint">{t.rag.topKHint}</span>
      </div>
      <div className="settings-row">
        <label>{t.rag.askSlot}</label>
        <select
          value={ragSlot}
          onChange={(e) => onRagSlotChange(e.target.value as KbSlot)}
        >
          <option value="resume">{t.rag.slotResume}</option>
          <option value="project">{t.rag.slotProject}</option>
          <option value="jd">{t.rag.slotJd}</option>
        </select>
        <span className="settings-inline-hint">{t.rag.askSlotHint}</span>
      </div>

      <details className="settings-advanced">
        <summary>{t.rag.advancedSection}</summary>
        <div className="settings-row">
          <label>{t.rag.neighborExpand}</label>
          {selectBool(neighborExpand, setNeighborExpand)}
          <span className="settings-inline-hint">{t.rag.neighborExpandHint}</span>
        </div>
        <div className="settings-row">
          <label>{t.rag.requireCitations}</label>
          {selectBool(requireCitations, setRequireCitations)}
          <span className="settings-inline-hint">{t.rag.requireCitationsHint}</span>
        </div>
      </details>

      {confirmWeak && (
        <div className="settings-warn">
          <div>{t.settings.weakCryptoWarning}</div>
          <div className="settings-actions" style={{ marginTop: 6 }}>
            <button className="btn btn-sm" onClick={() => setConfirmWeak(false)}>
              {t.settings.weakCryptoBack}
            </button>
            <button className="btn btn-sm" onClick={() => void save()}>
              {t.settings.weakCryptoContinue}
            </button>
          </div>
        </div>
      )}
      {saveError && <div className="settings-warn">{saveError}</div>}

      <div className="settings-actions">
        <span className="settings-inline-hint">
          {dirty ? t.rag.unsaved : savedAt ? t.rag.saved : ''}
        </span>
        <button className="btn btn-primary" disabled={saving} onClick={requestSave}>
          {saving ? t.rag.saving : t.rag.save}
        </button>
      </div>
    </>
  );
}
