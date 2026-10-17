/**
 * RAG 设置层的回归测试（MeetingCopilot 2.0）：
 *   * 默认值填充 —— 新装用户开箱即有 rag 配置（默认启用，但没 key 时必然低置信）；
 *   * 向后兼容 —— 老 settings.json（v2 无 rag 字段 / v1 迁移）必须能正常加载；
 *   * publicSettings() 必含 rag（PublicSettings.rag 是必填字段，缺了 UI 就崩）；
 *   * key hint 只暴露末 4 位，明文既不进 getPublic() 也不落盘；
 *   * 运行时配置（getRagConfig）是快照，改不动落盘数据。
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  SettingsStore,
  apiKeyHint,
  defaultRagSettings,
  defaultSettings,
  plainCipher,
  type SecretCipher,
} from '../electron/settings';
import { DEFAULT_RAG_CONFIG } from '../electron/rag';
import type { SettingsFile } from '../shared/protocol';

/**
 * 测试专用：settings.ts 内部把落盘槽扩展出一个 apiKeyEnc 字段（protocol 已锁定，
 * 未声明该字段），这里用同样的形状读/写密文，断言"密文在盘上、明文不在"。
 */
function encSlot(data: SettingsFile): { apiKeyEnc?: string } {
  return data.rag!.embedding as unknown as { apiKeyEnc?: string };
}

const fakeCipher: SecretCipher = {
  available: () => true,
  secure: true,
  encrypt: (plain) => `enc:${Buffer.from(plain, 'utf8').toString('base64')}`,
  decrypt: (s) => (s.startsWith('enc:') ? Buffer.from(s.slice(4), 'base64').toString('utf8') : ''),
};

let dir: string;
let file: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'mc-rag-settings-'));
  file = join(dir, 'settings.json');
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('RAG 默认值', () => {
  it('新装配置直接带 rag，取值来自 DEFAULT_RAG_CONFIG', () => {
    const s = new SettingsStore(file, fakeCipher);
    const rag = s.data.rag;
    expect(rag).toBeDefined();
    expect(rag).toEqual(defaultRagSettings());
    expect(rag!.enabled).toBe(DEFAULT_RAG_CONFIG.enabled);
    expect(rag!.topK).toBe(DEFAULT_RAG_CONFIG.topK);
    expect(rag!.minScore).toBe(DEFAULT_RAG_CONFIG.minScore);
    expect(rag!.neighborExpand).toBe(DEFAULT_RAG_CONFIG.neighborExpand);
    expect(rag!.requireCitations).toBe(DEFAULT_RAG_CONFIG.requireCitations);
    // 默认 embedding = 百炼 text-embedding-v3（与实时转写共用同一个 DashScope key）
    expect(rag!.embedding).toEqual(DEFAULT_RAG_CONFIG.embedding);
    expect(rag!.embedding.model).toBe('text-embedding-v3');
    expect(rag!.embedding.dim).toBe(1024);
  });

  it('defaultRagSettings() 返回副本，改它不影响 DEFAULT_RAG_CONFIG', () => {
    const a = defaultRagSettings();
    a.embedding.model = '被改坏了';
    a.topK = 99;
    expect(DEFAULT_RAG_CONFIG.embedding.model).toBe('text-embedding-v3');
    expect(DEFAULT_RAG_CONFIG.topK).toBe(5);
    expect(defaultRagSettings().embedding.model).toBe('text-embedding-v3');
  });

  it('getRagConfig() 是快照：返回 RagRuntimeConfig，且不外泄落盘对象', () => {
    const s = new SettingsStore(file, fakeCipher);
    const cfg = s.getRagConfig();
    expect(cfg).toEqual({
      enabled: DEFAULT_RAG_CONFIG.enabled,
      topK: DEFAULT_RAG_CONFIG.topK,
      minScore: DEFAULT_RAG_CONFIG.minScore,
      neighborExpand: DEFAULT_RAG_CONFIG.neighborExpand,
      requireCitations: DEFAULT_RAG_CONFIG.requireCitations,
      embedding: DEFAULT_RAG_CONFIG.embedding,
    });
    cfg.topK = 42;
    cfg.embedding.model = 'x';
    expect(s.data.rag!.topK).toBe(DEFAULT_RAG_CONFIG.topK);
    expect(s.data.rag!.embedding.model).toBe('text-embedding-v3');
  });
});

describe('向后兼容：老配置没有 rag 字段', () => {
  it('v2 文件（无 rag 段）加载后自动补默认值，其它字段不受影响', () => {
    writeFileSync(
      file,
      JSON.stringify({
        version: 2,
        onboarding: { schemaVersion: 1, completed: true },
        llm: { baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
      }),
      'utf8',
    );
    const s = new SettingsStore(file, fakeCipher);
    expect(s.migratedFromV1).toBe(false);
    expect(s.data.rag).toEqual(defaultRagSettings());
    // 老字段照旧——但 DeepSeek 旧模型名会被定向升级：
    // deepseek-chat 现在被官方静默转给 deepseek-flash（默认开思考模式），
    // 实测会让 max_tokens=300 时正文输出 0 字，所以加载时换成现行名并关思考。
    expect(s.data.llm.model).toBe('deepseek-flash');
    expect(s.data.llm.reasoningEffort).toBe('none');
    expect(s.data.llm.baseUrl).toBe('https://api.deepseek.com/v1');
    expect(s.data.onboarding.completed).toBe(true);
    expect(s.getPublic().rag.enabled).toBe(DEFAULT_RAG_CONFIG.enabled);
  });

  it('v2 文件只有半个 rag 段时，缺的字段用默认值补齐（不是 undefined）', () => {
    writeFileSync(
      file,
      JSON.stringify({ version: 2, rag: { enabled: false, embedding: { model: 'BAAI/bge-m3' } } }),
      'utf8',
    );
    const s = new SettingsStore(file, fakeCipher);
    const rag = s.data.rag!;
    expect(rag.enabled).toBe(false); // 用户的选择优先
    expect(rag.embedding.model).toBe('BAAI/bge-m3'); // 存下来的优先
    expect(rag.embedding.baseUrl).toBe(DEFAULT_RAG_CONFIG.embedding.baseUrl); // 补齐
    expect(rag.embedding.providerId).toBe('dashscope');
    expect(rag.topK).toBe(DEFAULT_RAG_CONFIG.topK);
    expect(rag.minScore).toBe(DEFAULT_RAG_CONFIG.minScore);
    expect(rag.requireCitations).toBe(DEFAULT_RAG_CONFIG.requireCitations);
  });

  it('v1 文件迁移到 v2 后同样带 rag 默认值', () => {
    writeFileSync(
      file,
      JSON.stringify({ llm: { baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' } }),
      'utf8',
    );
    const s = new SettingsStore(file, fakeCipher);
    expect(s.migratedFromV1).toBe(true);
    expect(s.data.rag).toEqual(defaultRagSettings());
    expect(s.getPublic().onboarding.migratedFromV1).toBe(true);
  });

  it('没有 rag 字段的库 boot 不写盘也不会崩（首次由向导决定文件长相）', () => {
    const s = new SettingsStore(file, fakeCipher);
    expect(s.data).toEqual(defaultSettings());
    expect(existsSync(file)).toBe(false);
  });
});

describe('publicSettings(): rag 投影', () => {
  it('rag 段字段齐全，未配 key 时 apiKeySet=false 且没有 hint', () => {
    const s = new SettingsStore(file, fakeCipher);
    const pub = s.getPublic();
    expect(pub.rag).toBeDefined();
    expect(pub.rag.enabled).toBe(true);
    expect(pub.rag.topK).toBe(DEFAULT_RAG_CONFIG.topK);
    expect(pub.rag.minScore).toBe(DEFAULT_RAG_CONFIG.minScore);
    expect(pub.rag.neighborExpand).toBe(true);
    expect(pub.rag.requireCitations).toBe(true);
    expect(pub.rag.embedding.baseUrl).toBe(DEFAULT_RAG_CONFIG.embedding.baseUrl);
    expect(pub.rag.embedding.model).toBe('text-embedding-v3');
    expect(pub.rag.embedding.dim).toBe(1024);
    expect(pub.rag.embedding.apiKeySet).toBe(false);
    expect(pub.rag.embedding.apiKeyHint).toBeUndefined();
  });

  it('patch 写入的 rag 值会出现在 publicSettings() 里（UI 回读一致）', () => {
    const s = new SettingsStore(file, fakeCipher);
    s.applyPatch({
      rag: {
        enabled: false,
        topK: 8,
        minScore: 0.62,
        neighborExpand: false,
        requireCitations: false,
        embedding: { providerId: 'siliconflow', baseUrl: 'https://api.siliconflow.cn/v1', model: 'BAAI/bge-m3', dim: 1024, batchSize: 8 },
      },
    });
    const pub = s.getPublic();
    expect(pub.rag.enabled).toBe(false);
    expect(pub.rag.topK).toBe(8);
    expect(pub.rag.minScore).toBe(0.62);
    expect(pub.rag.neighborExpand).toBe(false);
    expect(pub.rag.requireCitations).toBe(false);
    expect(pub.rag.embedding.providerId).toBe('siliconflow');
    expect(pub.rag.embedding.model).toBe('BAAI/bge-m3');
    expect(pub.rag.embedding.batchSize).toBe(8);
    // 落盘同样生效（进程重启后一致）
    const again = new SettingsStore(file, fakeCipher).getPublic();
    expect(again.rag).toEqual(pub.rag);
  });

  it('rag 关闭后 getRagConfig().enabled=false，主进程据此退回老路径', () => {
    const s = new SettingsStore(file, fakeCipher);
    s.applyPatch({ rag: { enabled: false } });
    expect(s.getRagConfig().enabled).toBe(false);
    expect(s.getPublic().rag.enabled).toBe(false);
  });
});

describe('embedding Key：只露末 4 位，明文不落盘', () => {
  const KEY = 'sk-embed-abcd1234';

  it('applyPatch 保存 key：密文进 settings.json，hint 只有末 4 位', () => {
    const s = new SettingsStore(file, fakeCipher);
    // 只送 {apiKey}：契约（SettingsPatch.rag.embedding 为 Partial）必须允许 UI 这么写
    s.applyPatch({ rag: { embedding: { apiKey: KEY } } });

    expect(s.getEmbeddingApiKey()).toBe(KEY);
    expect(s.data.rag!.embedding.apiKeyHint).toBe(KEY.slice(-4));
    expect(apiKeyHint(KEY)).toBe('1234');
    expect(s.getPublic().rag.embedding.apiKeySet).toBe(true);
    expect(s.getPublic().rag.embedding.apiKeyHint).toBe('1234');

    // 明文绝不出现在对外结构里（含完整 key 的 JSON 序列化）
    const pubJson = JSON.stringify(s.getPublic());
    expect(pubJson).not.toContain(KEY);
    expect(pubJson).not.toContain('abcd1234');
    // 也不落盘（只有 cipher.encrypt 后的密文）
    const onDisk = readFileSync(file, 'utf8');
    expect(onDisk).not.toContain(KEY);
    expect(onDisk).not.toContain('abcd1234');
    expect(onDisk).toContain(fakeCipher.encrypt(KEY));
  });

  it('setEmbeddingApiKey 与 LLM key 用同一套加密机制（同一 cipher）', () => {
    const s = new SettingsStore(file, fakeCipher);
    s.applyPatch({ llm: { apiKey: 'sk-llm-9999' } });
    s.setEmbeddingApiKey(KEY);
    // 2026-04-17 之类的短 key 也走同一密文格式
    expect(s.getLlmApiKey()).toBe('sk-llm-9999');
    expect(s.getEmbeddingApiKey()).toBe(KEY);
    expect(s.data.rag!.embedding.apiKeyHint).toBe('1234');
    expect(s.getPublic().llm.apiKeyHint).toBe('9999');
    // 两个槽互不串味
    expect(s.data.llm.apiKeyEnc).not.toBe(encSlot(s.data).apiKeyEnc);
  });

  it('plainCipher（无 OS 加密时的兜底）同样是密文形态、仍只露末 4 位', () => {
    const s = new SettingsStore(file, plainCipher);
    s.setEmbeddingApiKey('sk-plain-key-7788');
    expect(s.getPublic().weakCrypto).toBe(true);
    expect(s.getEmbeddingApiKey()).toBe('sk-plain-key-7788');
    expect(s.getPublic().rag.embedding.apiKeyHint).toBe('7788');
    expect(readFileSync(file, 'utf8')).not.toContain('sk-plain-key-7788');
  });

  it('空字符串清除该槽；hint 一并清掉', () => {
    const s = new SettingsStore(file, fakeCipher);
    s.setEmbeddingApiKey(KEY);
    s.setEmbeddingApiKey('');
    expect(s.getEmbeddingApiKey()).toBeUndefined();
    expect(s.getPublic().rag.embedding.apiKeySet).toBe(false);
    expect(s.getPublic().rag.embedding.apiKeyHint).toBeUndefined();
    expect(readFileSync(file, 'utf8')).not.toContain(fakeCipher.encrypt(KEY));
  });

  it('只改 baseUrl/model 不会把已存的 key 冲掉', () => {
    const s = new SettingsStore(file, fakeCipher);
    s.setEmbeddingApiKey(KEY);
    // 只改模型（不带 key/baseUrl）：partial 契约下必须合法，且不得冲掉已存的 key
    s.applyPatch({ rag: { embedding: { model: 'text-embedding-v4' } } });
    expect(s.getEmbeddingApiKey()).toBe(KEY);
    expect(s.getPublic().rag.embedding.model).toBe('text-embedding-v4');
    expect(s.getPublic().rag.embedding.apiKeyHint).toBe('1234');
  });

  it('密文损坏时读 key 返回 undefined，不让主进程抛异常', () => {
    // safeStorage 解不开时是**抛异常**（换机器 / 换 OS 账户），用同形态的 cipher 复现
    const strictCipher: SecretCipher = {
      available: () => true,
      secure: true,
      encrypt: fakeCipher.encrypt,
      decrypt: (b64) => {
        if (!b64.startsWith('enc:')) throw new Error('Error while decrypting the ciphertext');
        return Buffer.from(b64.slice(4), 'base64').toString('utf8');
      },
    };
    const s = new SettingsStore(file, strictCipher);
    s.setEmbeddingApiKey(KEY);
    encSlot(s.data).apiKeyEnc = 'not-a-valid-ciphertext';
    expect(() => s.getEmbeddingApiKey()).not.toThrow();
    expect(s.getEmbeddingApiKey()).toBeUndefined();
    expect(s.getPublic().rag.embedding.apiKeySet).toBe(true); // 密文在，只是解不开
  });
});
