import type { CitationAudit, FactLevel, RagEvidence, RagRetrieval } from '../../shared/protocol';
import { useT } from '../i18n';

/**
 * 与某条回答关联的检索结果。由 App 在 `onLlmEvent` 的
 * `kind === 'retrieval'` 分支里按 `requestId` 收集，仅存于内存
 * （StoredTurn 的字段在 shared/protocol.ts 里锁定，不能扩展；
 * 重启后「依据」区块消失，属预期行为）。
 */
export interface TurnRetrieval {
  retrieval: RagRetrieval;
  /** true = 资料里没找到依据，本次是固定兜底话术，**没有**调用大模型 */
  usedFallback: boolean;
  /**
   * 引用覆盖率审计。答案完成后由主进程以 `kind: 'citation'` 单独推送
   * （必须等回答完整才能核对），所以在 retrieval 到达时通常是 undefined。
   */
  audit?: CitationAudit;
}

/** 事实档位徽章：verified 绿 · claimed 橙（警示色）· unverified 灰 */
export function FactBadge({ level }: { level: FactLevel }) {
  const t = useT();
  const cls =
    level === 'verified' ? 'tag tag-ok' : level === 'claimed' ? 'tag tag-claimed' : 'tag';
  return (
    <span className={cls} title={t.answer.factTip[level]}>
      {t.answer.factLevel[level]}
    </span>
  );
}

/** 命中正文摘要（折叠态只给前 N 个字符，悬浮可见全文） */
export function excerpt(text: string, max: number): string {
  const one = text.replace(/\s+/g, ' ').trim();
  return one.length > max ? `${one.slice(0, max)}…` : one;
}

/**
 * 一条证据：`[S1] 来源 · 标题 · 事实档位 · 原始余弦` + 正文前 N 字。
 * 答案区的「依据」与设置面板的「检索自测」共用。
 */
export function EvidenceRow({
  ev,
  minScore,
  chars,
  showBonus = false,
}: {
  ev: RagEvidence;
  /** 低于该原始余弦视为「资料里没找到」，分数标灰 */
  minScore: number;
  chars: number;
  showBonus?: boolean;
}) {
  const t = useT();
  const low = ev.score < minScore;
  const source = ev.heading ? `${ev.source} · ${ev.heading}` : ev.source;
  return (
    <li className="rag-evidence-item">
      <div className="rag-evidence-head">
        <span className="rag-ref">[{ev.ref}]</span>
        <span className="rag-source" title={source}>
          {source}
        </span>
        <FactBadge level={ev.factLevel} />
        <span className={low ? 'rag-score rag-score-low' : 'rag-score'} title={t.rag.minScore}>
          {ev.score.toFixed(3)}
        </span>
        {showBonus && ev.keywordBonus > 0 && (
          <span className="rag-bonus">{t.answer.keywordBonus(ev.keywordBonus.toFixed(2))}</span>
        )}
      </div>
      <div className="rag-evidence-text" title={ev.text}>
        {excerpt(ev.text, chars)}
      </div>
    </li>
  );
}

/**
 * 引用覆盖率审计区块：回答里哪些事实句没有出处。
 *
 * 这是"防瞎编"的最后一环，也是**唯一能当场自查**的一环：
 * 前面几层只能"尽量别让模型编"，这里直接把"哪句没依据"摆出来，
 * 让你在念出口之前自己核一眼。
 */
export function CitationAuditBlock({ audit }: { audit: CitationAudit }) {
  const t = useT();
  const pct = Math.round(audit.coverage * 100);
  const hasProblem = audit.uncitedFacts.length > 0 || audit.dangling.length > 0;
  const citedCount = audit.factCount - audit.uncitedFacts.length;

  // 没有事实句：不算问题，也不必占版面
  if (audit.factCount === 0) {
    return <div className="rag-cite rag-cite-ok">{t.answer.citeNoFacts}</div>;
  }

  return (
    <div className={hasProblem ? 'rag-cite rag-cite-warn' : 'rag-cite rag-cite-ok'}>
      <div className="rag-cite-head">
        {hasProblem
          ? t.answer.citeCoverageWarn(audit.uncitedFacts.length)
          : t.answer.citePerfect}
        <span className="rag-cite-pct">
          {t.answer.citeCoverage(pct, citedCount, audit.factCount)}
        </span>
      </div>
      {audit.dangling.length > 0 && (
        <div className="rag-cite-dangling">{t.answer.citeDangling(audit.dangling.join('、'))}</div>
      )}
      {audit.uncitedFacts.length > 0 && (
        <details className="rag-cite-details" open>
          <summary>{t.answer.citeUncitedTitle}</summary>
          <ul className="rag-cite-list">
            {audit.uncitedFacts.slice(0, 12).map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/**
 * 回答下方的「依据」区块（引用溯源）。折叠展示本次回答实际参考的证据，
 * 让用户能核对 AI 到底依据了什么；走兜底话术时用琥珀色横幅显著标注。
 */
export function RagEvidenceBlock({
  data,
  minScore,
}: {
  data: TurnRetrieval;
  /** 低置信阈值（原始余弦），用于标注「分数不够」的命中 */
  minScore: number;
}) {
  const t = useT();
  const { retrieval, usedFallback, audit } = data;
  const evidence = retrieval.evidence;
  const top = retrieval.topScore || evidence[0]?.score || 0;
  return (
    <div className="rag-evidence-wrap">
      {usedFallback && (
        <div className="rag-fallback">
          <span className="rag-fallback-title">{t.answer.evidenceFallback}</span>
          <div className="rag-fallback-hint">{t.answer.evidenceFallbackHint}</div>
        </div>
      )}
      {/* 兜底话术是固定文本、本来就没有引用编号，审计它只会产生噪声 */}
      {audit && !usedFallback && <CitationAuditBlock audit={audit} />}
      <details className="rag-evidence">
        <summary className="rag-evidence-summary">
          <span className="rag-evidence-title">{t.answer.evidence}</span>
          <span className="rag-evidence-meta">
            {t.answer.evidenceHits(evidence.length)} · {t.answer.evidenceTop(top.toFixed(3))} ·{' '}
            {t.answer.evidenceRetrievalMs(retrieval.latencyMs)}
          </span>
        </summary>
        {evidence.length === 0 ? (
          <div className="rag-evidence-note">{t.answer.evidenceNone}</div>
        ) : (
          <ul className="rag-evidence-list">
            {evidence.map((ev) => (
              <EvidenceRow
                key={`${ev.ref}-${ev.chunkId}`}
                ev={ev}
                minScore={minScore}
                chars={100}
              />
            ))}
          </ul>
        )}
        {!usedFallback && retrieval.lowConfidence && (
          <div className="rag-evidence-note is-warn">
            {t.answer.evidenceLowNote(minScore.toFixed(2))}
          </div>
        )}
        <div className="rag-evidence-note">{t.answer.evidenceScoreNote(minScore.toFixed(2))}</div>
      </details>
    </div>
  );
}
