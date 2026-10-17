import { contextBridge, ipcRenderer } from 'electron';
import {
  IPC,
  type AppInfo,
  type AsrEvent,
  type KbPickResult,
  type KbSlot,
  type LlmAskPayload,
  type LlmEvent,
  type OnboardingProgressPatch,
  type OnboardingState,
  type ProviderTestRequest,
  type ProviderTestResult,
  type PublicSettings,
  type RagIndexInfo,
  type RagIndexProgress,
  type RagRetrieval,
  type SessionsFile,
  type SettingsPatch,
  type TrayCommandPayload,
} from '../shared/protocol';

/** 索引类 IPC 的返回：成功给总览，失败给中文原因（主进程已兜住异常，UI 只需展示） */
export type RagIndexOutcome = RagIndexInfo | { error: string };
/** 检索自测的返回：成功给命中列表（含分数），失败给中文原因 */
export type RagSearchOutcome = RagRetrieval | { error: string };

/** embedding 连通性测试的入参；apiKey 仅本次使用，绝不落盘 */
export interface RagEmbeddingTestRequest {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  dim?: number;
}

/** 与 electron/main.ts 的 IPC.ragEmbeddingTest 返回一致（testEmbedding 的形状） */
export interface RagEmbeddingTestResult {
  ok: boolean;
  messageZh: string;
  dim?: number;
  latencyMs?: number;
}

export interface McApi {
  readonly platform: NodeJS.Platform;
  getSettings(): Promise<PublicSettings>;
  setSettings(patch: SettingsPatch): Promise<PublicSettings>;
  /** first-run wizard state (the main window only reads it / dismisses the
   * upgrade notice — completing onboarding belongs to the setup window) */
  getOnboarding(): Promise<OnboardingState>;
  saveOnboardingProgress(patch: OnboardingProgressPatch): Promise<OnboardingState>;
  /** reopen the setup wizard without closing the app (settings / upgrade notice) */
  rerunOnboarding(): Promise<boolean>;
  importKnowledge(): Promise<{ chars: number }>;
  clearKnowledge(): Promise<{ chars: number }>;
  /** 为某个槽位挑文档，**支持多选**（.md/.txt/.docx/.pdf）；返回成功解析的文档与逐文件失败原因 */
  pickKnowledge(slot: KbSlot): Promise<KbPickResult>;
  /** RAG（MeetingCopilot 2.0）：索引总览（片段数 / 来源 / 模型 / 维度） */
  ragIndexGet(): Promise<RagIndexInfo>;
  /** 用已解析的文本重建某个槽位的索引（embedding 调用在 main 侧，失败返回 {error}）*/
  ragIndexRebuild(req: { slot: KbSlot; source: string; text: string }): Promise<RagIndexOutcome>;
  /** 清空全部索引（原始资料不动） */
  ragIndexClear(): Promise<RagIndexInfo>;
  /** 检索自测：看命中了什么、原始分数多少（minScore 不过滤，便于校准阈值）*/
  ragSearchTest(req: { query: string; topK?: number }): Promise<RagSearchOutcome>;
  /** 测试 embedding 端点；传了 apiKey 就用临时 key 测（只此一次，不落盘）*/
  ragEmbeddingTest(req?: RagEmbeddingTestRequest): Promise<RagEmbeddingTestResult>;
  /** 索引进度（流式，main -> renderer）；返回取消订阅函数 */
  onRagProgress(cb: (p: RagIndexProgress) => void): () => void;
  loadSessions(): Promise<SessionsFile>;
  saveSessions(data: SessionsFile): void;
  setStealth(on: boolean): Promise<boolean>;
  sendPcm(buf: ArrayBuffer, captureTs: number, channel: 'them' | 'me'): void;
  captureStarted(): void;
  captureStopped(): void;
  translate(text: string): Promise<string>;
  onAsrEvent(cb: (ev: AsrEvent) => void): () => void;
  /** pull the last ready/status events (call AFTER onAsrEvent subscription) */
  asrReplay(): Promise<{ ready: AsrEvent | null; status: AsrEvent | null }>;
  llmAsk(payload: LlmAskPayload): void;
  shotAsk(payload: {
    requestId: string;
    question: string;
    background?: string;
    imageDataUrl?: string;
  }): void;
  /** capture full screen, drag a stealth region overlay; returns cropped dataURL or null */
  pickRegion(): Promise<string | null>;
  /** overlay-only: fetch the captured background image */
  regionImage(): Promise<string | null>;
  /** overlay-only: report chosen rect */
  regionRect(r: { x: number; y: number; width: number; height: number }): void;
  /** overlay-only: cancel */
  regionCancel(): void;
  llmCancel(requestId: string): void;
  /** P1-6: warm the DeepSeek prefix cache with the session's material;
   * immediate=true warms even when not capturing (▶ start / material import) */
  prewarm(payload: { resume?: string; jd?: string; immediate?: boolean }): void;
  /** P1-5: fold a finished Q&A into the rolling memo ('' = keep the old one) */
  memoUpdate(p: { memo: string; question: string; answer: string }): Promise<string>;
  onLlmEvent(cb: (ev: LlmEvent) => void): () => void;
  onShotHotkey(cb: () => void): () => void;
  /** tray menu entries only the renderer can service (capture / session /
   * panels). Main has already made the window visible when this fires. */
  onTrayCommand(cb: (payload: TrayCommandPayload) => void): () => void;
  /** open an allowlisted https documentation link in the OS browser;
   * false = refused by the main-process allowlist */
  openExternal(url: string): Promise<boolean>;
  /** read the clipboard — call ONLY from an explicit paste-button click */
  readClipboardText(): Promise<string>;
  /** PNG dataURL of the clipboard image — call ONLY from an explicit Ctrl+V
   * paste in the answer input */
  readClipboardImage(): Promise<string | null>;
  getAppInfo(): Promise<AppInfo>;
  /** run ONE real provider connection test; explicit user action only (it
   * costs the user a minimal billable request). Main maps every failure to a
   * ProviderTestCode + zh/en message, so nothing raw reaches the UI. */
  providerTest(req: ProviderTestRequest): Promise<ProviderTestResult>;
  /** plaintext support report, built locally on request. Contains no keys, no
   * transcripts and no knowledge-base text — safe to paste into an issue. */
  getDiagnostics(): Promise<string>;
  /** reveal %APPDATA%/MeetingCopilot (settings, sessions, knowledge) */
  openLogsFolder(): Promise<boolean>;
  hide(): void;
  quit(): void;
}

const api: McApi = {
  platform: process.platform,
  getSettings: () => ipcRenderer.invoke(IPC.settingsGet),
  setSettings: (patch) => ipcRenderer.invoke(IPC.settingsSet, patch),
  getOnboarding: () => ipcRenderer.invoke(IPC.onboardingGet),
  saveOnboardingProgress: (patch) => ipcRenderer.invoke(IPC.onboardingSaveProgress, patch),
  rerunOnboarding: () => ipcRenderer.invoke(IPC.onboardingRerun),
  importKnowledge: () => ipcRenderer.invoke(IPC.knowledgeImport),
  clearKnowledge: () => ipcRenderer.invoke(IPC.knowledgeClear),
  pickKnowledge: (slot) => ipcRenderer.invoke(IPC.knowledgePick, slot),
  ragIndexGet: () => ipcRenderer.invoke(IPC.ragIndexGet),
  ragIndexRebuild: (req) => ipcRenderer.invoke(IPC.ragIndexRebuild, req),
  ragIndexClear: () => ipcRenderer.invoke(IPC.ragIndexClear),
  ragSearchTest: (req) => ipcRenderer.invoke(IPC.ragSearchTest, req),
  ragEmbeddingTest: (req) => ipcRenderer.invoke(IPC.ragEmbeddingTest, req),
  onRagProgress: (cb) => {
    const listener = (_e: Electron.IpcRendererEvent, p: RagIndexProgress) => cb(p);
    ipcRenderer.on(IPC.ragProgress, listener);
    return () => ipcRenderer.removeListener(IPC.ragProgress, listener);
  },
  loadSessions: () => ipcRenderer.invoke(IPC.sessionsLoad),
  saveSessions: (data) => ipcRenderer.send(IPC.sessionsSave, data),
  setStealth: (on) => ipcRenderer.invoke(IPC.stealthSet, on),
  sendPcm: (buf, captureTs, channel) => ipcRenderer.send(IPC.capturePcm, buf, captureTs, channel),
  captureStarted: () => ipcRenderer.send(IPC.captureStarted),
  captureStopped: () => ipcRenderer.send(IPC.captureStopped),
  translate: (text) => ipcRenderer.invoke(IPC.translateText, text),
  onAsrEvent: (cb) => {
    const listener = (_e: Electron.IpcRendererEvent, ev: AsrEvent) => cb(ev);
    ipcRenderer.on(IPC.asrEvent, listener);
    return () => ipcRenderer.removeListener(IPC.asrEvent, listener);
  },
  asrReplay: () => ipcRenderer.invoke(IPC.asrReplay),
  llmAsk: (payload) => ipcRenderer.send(IPC.llmAsk, payload),
  shotAsk: (payload) => ipcRenderer.send(IPC.shotAsk, payload),
  pickRegion: () => ipcRenderer.invoke(IPC.regionPick),
  regionImage: () => ipcRenderer.invoke(IPC.regionImage),
  regionRect: (r) => ipcRenderer.send(IPC.regionRect, r),
  regionCancel: () => ipcRenderer.send(IPC.regionCancel),
  llmCancel: (requestId) => ipcRenderer.send(IPC.llmCancel, requestId),
  prewarm: (payload) => ipcRenderer.send(IPC.llmPrewarm, payload),
  memoUpdate: (p) => ipcRenderer.invoke(IPC.memoUpdate, p),
  onLlmEvent: (cb) => {
    const listener = (_e: Electron.IpcRendererEvent, ev: LlmEvent) => cb(ev);
    ipcRenderer.on(IPC.llmEvent, listener);
    return () => ipcRenderer.removeListener(IPC.llmEvent, listener);
  },
  onShotHotkey: (cb) => {
    const listener = () => cb();
    ipcRenderer.on(IPC.shotHotkey, listener);
    return () => ipcRenderer.removeListener(IPC.shotHotkey, listener);
  },
  onTrayCommand: (cb) => {
    const listener = (_e: Electron.IpcRendererEvent, payload: TrayCommandPayload) => cb(payload);
    ipcRenderer.on(IPC.trayCommand, listener);
    return () => ipcRenderer.removeListener(IPC.trayCommand, listener);
  },
  openExternal: (url) => ipcRenderer.invoke(IPC.externalOpen, url),
  readClipboardText: () => ipcRenderer.invoke(IPC.clipboardReadText),
  readClipboardImage: () => ipcRenderer.invoke(IPC.clipboardReadImage),
  getAppInfo: () => ipcRenderer.invoke(IPC.appGetInfo),
  providerTest: (req) => ipcRenderer.invoke(IPC.providerTest, req),
  getDiagnostics: () => ipcRenderer.invoke(IPC.diagnosticsGet),
  openLogsFolder: () => ipcRenderer.invoke(IPC.logsOpenFolder),
  hide: () => ipcRenderer.send(IPC.winHide),
  quit: () => ipcRenderer.send(IPC.appQuit),
};

contextBridge.exposeInMainWorld('mc', api);
