import { useEffect, useRef, useState, type ClipboardEvent } from 'react';
import type { KbDoc, KbSlot, StoredSession } from '../../shared/protocol';
import { useT } from '../i18n';
import { RagEvidenceBlock, type TurnRetrieval } from './RagEvidenceView';

export type TurnKind = 'segment' | 'continuous' | 'free' | 'translate' | 'vision';

/** read a pasted image File/blob as a dataURL */
function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

export interface AnswerTurn {
  id: string;
  kind: TurnKind;
  label: string;
  text: string;
  status: 'streaming' | 'done' | 'error';
  error?: string;
}

/**
 * 一个资料槽位的按钮：显示槽名与已导入份数，点开可看清单、逐份移除。
 *
 * 为什么要做成"按钮 + 下拉"而不是"按钮 + ×"：多文档之后，× 只能整槽清空，
 * 导错一份就得全部重来。清单里逐份移除才是多文档该有的操作。
 */
function SlotButton({
  slot,
  docs,
  onPick,
  onClear,
  onRemove,
}: {
  slot: KbSlot;
  docs: KbDoc[];
  onPick: (slot: KbSlot) => void;
  onClear: (slot: KbSlot) => void;
  onRemove: (slot: KbSlot, docId: string) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const n = docs.length;
  const chars = docs.reduce((s, d) => s + d.text.length, 0);

  // 点外面收起清单
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const label = slot === 'resume' ? t.answer.resume : slot === 'project' ? t.answer.project : t.answer.jd;
  const icon = slot === 'resume' ? '📄' : slot === 'project' ? '📁' : '📋';
  const title = n > 0 ? t.answer.slotSetTitle(label, n, chars) : t.answer.slotEmptyTitle(label);

  return (
    <span className="slot-wrap" ref={wrapRef}>
      <button
        className={n > 0 ? 'btn btn-sm btn-on' : 'btn btn-sm'}
        title={title}
        onClick={() => {
          // 有内容时先展开清单（可逐份移除 / 继续追加）；空槽直接开文件框
          if (n === 0) onPick(slot);
          else setOpen((v) => !v);
        }}
      >
        {icon}
        {label}
        {n > 1 && <span className="slot-count">{n}</span>}
      </button>
      {open && n > 0 && (
        <div className="slot-menu">
          <div className="slot-menu-head">{t.answer.slotMenuTitle(label, n)}</div>
          <ul className="slot-menu-list">
            {docs.map((d) => (
              <li key={d.id}>
                <span className="slot-doc-name" title={d.name}>
                  {d.name}
                </span>
                <span className="slot-doc-chars">{t.answer.docChars(d.text.length)}</span>
                <button
                  className="btn btn-sm"
                  title={t.answer.docRemoveTitle(d.name)}
                  onClick={() => onRemove(slot, d.id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          <div className="slot-menu-actions">
            <button
              className="btn btn-sm"
              onClick={() => {
                setOpen(false);
                onPick(slot);
              }}
            >
              {t.answer.slotAddMore}
            </button>
            <button
              className="btn btn-sm"
              onClick={() => {
                setOpen(false);
                onClear(slot);
              }}
            >
              {t.answer.slotClearAll}
            </button>
          </div>
        </div>
      )}
    </span>
  );
}

/**
 * 智能体 conversation panel (R4) with a multi-session bar. Answers ACCUMULATE
 * as a scrolling session (never replaced); each meeting is its own session
 * with its own knowledge base. The 📷 screenshot button only appears in
 * multimodal mode (it needs a vision model).
 */
export function AnswerSession({
  sessions,
  currentId,
  turns,
  docsBySlot,
  notice,
  visionReady,
  visionConfigured,
  answersReady,
  answersHint,
  evidence,
  ragMinScore,
  onSwitch,
  onNew,
  onDelete,
  onRename,
  onPickKb,
  onClearKb,
  onRemoveKbDoc,
  onCancel,
  onClear,
  onFreeAsk,
  onShotAsk,
}: {
  sessions: StoredSession[];
  currentId: string;
  turns: AnswerTurn[];
  /** 三个槽位各自已导入的文档（简历 / 项目 / 岗位 JD） */
  docsBySlot: Record<KbSlot, KbDoc[]>;
  /** transient parse warning (e.g. scanned PDF with no text layer) */
  notice?: string | null;
  visionReady: boolean;
  /** a vision provider is configured (independent of the Text/Vision toggle):
   * enables pasting a screenshot from the clipboard into the input */
  visionConfigured: boolean;
  /** false = no LLM configured; asking is disabled with an explanation */
  answersReady: boolean;
  answersHint: string;
  /** requestId → 该次回答的检索结果（引用溯源，仅存于内存） */
  evidence: Record<string, TurnRetrieval>;
  /** 低置信阈值（原始余弦），用于「依据」区块里分数是否达标的标注 */
  ragMinScore: number;
  onSwitch: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onPickKb: (slot: KbSlot) => void;
  onClearKb: (slot: KbSlot) => void;
  /** 移除某个槽位里的单份文档（多文档场景必备：导错一份不用清空整个槽） */
  onRemoveKbDoc: (slot: KbSlot, docId: string) => void;
  onCancel: (id: string) => void;
  onClear: () => void;
  onFreeAsk: (question: string) => void;
  onShotAsk: (question: string, imageDataUrl?: string) => void;
}) {
  const t = useT();
  const boxRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const inputRef = useRef<HTMLInputElement>(null);
  const [editing, setEditing] = useState(false);
  const [nameDraft, setNameDraft] = useState('');
  /** a pasted screenshot (Win+Shift+S → Ctrl+V) waiting in the input */
  const [pendingImage, setPendingImage] = useState<string | null>(null);
  const currentName = sessions.find((s) => s.id === currentId)?.name ?? '';

  useEffect(() => {
    if (stick.current && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  });

  const onScroll = () => {
    const el = boxRef.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const submit = () => {
    const q = inputRef.current?.value.trim() ?? '';
    if (!q && !pendingImage) return;
    if (pendingImage) {
      // the pasted screenshot goes to the vision model; the typed text is the question
      onShotAsk(q, pendingImage);
      setPendingImage(null);
    } else {
      onFreeAsk(q);
    }
    if (inputRef.current) inputRef.current.value = '';
  };

  /**
   * Win+Shift+S puts the snip on the OS clipboard; Ctrl+V here picks it up.
   * A DOM image item is preferred (works cross-platform); if the browser
   * exposes no image and there is no text either, fall back to reading the
   * clipboard image in the main process (Windows CF_DIB). Never polled.
   */
  const onPaste = async (e: ClipboardEvent<HTMLInputElement>) => {
    if (!visionConfigured) return; // let the default paste behave normally
    const items = e.clipboardData?.items;
    let file: File | null = null;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          file = it.getAsFile();
          break;
        }
      }
    }
    if (file) {
      e.preventDefault();
      setPendingImage(await blobToDataUrl(file));
      return;
    }
    if (!e.clipboardData?.getData('text/plain')) {
      e.preventDefault();
      const img = await window.mc.readClipboardImage();
      if (img) setPendingImage(img);
    }
  };

  return (
    <section className="pane pane-answer">
      <header className="pane-head session-bar">
        {editing ? (
          <input
            className="session-select"
            autoFocus
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={() => {
              onRename(currentId, nameDraft);
              setEditing(false);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                onRename(currentId, nameDraft);
                setEditing(false);
              } else if (e.key === 'Escape') {
                setEditing(false);
              }
            }}
          />
        ) : (
          <select
            className="session-select"
            value={currentId}
            onChange={(e) => onSwitch(e.target.value)}
            title={t.answer.switchTitle}
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        )}
        <button
          className="btn btn-sm"
          onClick={() => {
            setNameDraft(currentName);
            setEditing(true);
          }}
          title={t.answer.renameTitle}
        >
          ✎
        </button>
        <button className="btn btn-sm" onClick={onNew} title={t.answer.newTitle}>
          ＋
        </button>
        <button className="btn btn-sm" onClick={() => onDelete(currentId)} title={t.answer.deleteTitle}>
          🗑
        </button>
        <SlotButton
          slot="resume"
          docs={docsBySlot.resume}
          onPick={onPickKb}
          onClear={onClearKb}
          onRemove={onRemoveKbDoc}
        />
        <SlotButton
          slot="project"
          docs={docsBySlot.project}
          onPick={onPickKb}
          onClear={onClearKb}
          onRemove={onRemoveKbDoc}
        />
        <SlotButton
          slot="jd"
          docs={docsBySlot.jd}
          onPick={onPickKb}
          onClear={onClearKb}
          onRemove={onRemoveKbDoc}
        />
        <span className="session-spacer" />
        <button className="btn btn-sm" onClick={onClear} title={t.answer.clearTitle}>
          {t.answer.clear}
        </button>
      </header>
      {notice && <div className="kb-notice">{notice}</div>}
      <div className="session" ref={boxRef} onScroll={onScroll}>
        {turns.length === 0 ? (
          <div className="pane-empty">
            {t.answer.empty}
            {/* 三个槽位全空时才提示怎么导入资料 */}
            {Object.values(docsBySlot).every((d) => d.length === 0) && t.answer.emptyKbHint}
          </div>
        ) : (
          turns.map((turn) => (
            <div key={turn.id} className={`turn turn-${turn.kind}`}>
              <div className="turn-head">
                <span className="turn-tag">{t.answer.kindTag[turn.kind]}</span>
                <span className="turn-label" title={turn.label}>
                  {turn.label}
                </span>
                {turn.status === 'streaming' ? (
                  <button className="btn btn-sm" onClick={() => onCancel(turn.id)}>
                    {t.answer.stop}
                  </button>
                ) : (
                  <button
                    className="btn btn-sm"
                    onClick={() => void navigator.clipboard.writeText(turn.text)}
                    title={t.answer.copyTitle}
                  >
                    {t.answer.copy}
                  </button>
                )}
              </div>
              <div className="turn-body">
                {turn.status === 'error' ? (
                  <span className="answer-error">{turn.error}</span>
                ) : (
                  <>
                    {turn.text || (turn.kind === 'vision' ? t.answer.visionWaiting : t.answer.genWaiting)}
                    {turn.status === 'streaming' && <span className="cursor">▍</span>}
                  </>
                )}
              </div>
              {/* 引用溯源：本次回答到底参考了哪些资料片段（走兜底时显著标注） */}
              {evidence[turn.id] && (
                <RagEvidenceBlock data={evidence[turn.id]} minScore={ragMinScore} />
              )}
            </div>
          ))
        )}
      </div>
      {!answersReady && <div className="kb-notice">{answersHint}</div>}
      {pendingImage && (
        <div className="answer-attach">
          <img src={pendingImage} alt={t.answer.attachAlt} />
          <span className="answer-attach-hint">{t.answer.attachHint}</span>
          <button
            className="btn btn-sm"
            onClick={() => setPendingImage(null)}
            title={t.answer.attachRemove}
          >
            ×
          </button>
        </div>
      )}
      <div className="answer-input">
        <input
          ref={inputRef}
          disabled={!answersReady}
          onPaste={(e) => void onPaste(e)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) submit();
          }}
          placeholder={
            !answersReady
              ? answersHint
              : pendingImage
                ? t.answer.attachPlaceholder
                : t.answer.freePlaceholder
          }
        />
        <button
          className="btn btn-primary"
          disabled={!answersReady}
          title={answersReady ? undefined : answersHint}
          onClick={submit}
        >
          {t.answer.ask}
        </button>
        {visionReady && (
          <button
            className="btn"
            title={t.answer.shotTitle}
            onClick={async () => {
              const q = inputRef.current?.value.trim() ?? '';
              const img = await window.mc.pickRegion();
              if (!img) return; // cancelled
              if (inputRef.current) inputRef.current.value = '';
              onShotAsk(q, img);
            }}
          >
            📷
          </button>
        )}
      </div>
    </section>
  );
}
