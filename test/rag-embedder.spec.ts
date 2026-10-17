/**
 * Embedding 客户端单测：端点规范化、批量正确性、错误分类重试策略。
 * 用 mock fetch，不联网。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { embedTexts, embeddingsEndpoint, testEmbedding } from '../electron/rag/embedder';

describe('embeddingsEndpoint 规范化', () => {
  it('接受各种用户输入的 baseUrl', () => {
    expect(embeddingsEndpoint('https://dashscope.aliyuncs.com/compatible-mode/v1')).toBe(
      'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
    );
    expect(embeddingsEndpoint('https://api.siliconflow.cn/v1')).toBe('https://api.siliconflow.cn/v1/embeddings');
    expect(embeddingsEndpoint('https://api.openai.com/v1/')).toBe('https://api.openai.com/v1/embeddings');
    // 已到 /embeddings 不重复追加
    expect(embeddingsEndpoint('https://x/v1/embeddings')).toBe('https://x/v1/embeddings');
    // 只填域名 → 补 /v1/embeddings
    expect(embeddingsEndpoint('https://example.com')).toBe('https://example.com/v1/embeddings');
  });
  it('空 baseUrl 抛错', () => {
    expect(() => embeddingsEndpoint('   ')).toThrow(/为空/);
  });
});

const settings = { providerId: 'dashscope', baseUrl: 'https://x/v1', model: 'm', dim: 1024, batchSize: 2 };

function okResponse(vectors: number[][], order?: number[]) {
  const data = vectors.map((embedding, i) => ({ embedding, index: order ? order[i] : i }));
  return { ok: true, status: 200, text: async () => JSON.stringify({ data, usage: { total_tokens: 7 } }) };
}

describe('embedTexts', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.restoreAllMocks());

  it('空输入不发请求', async () => {
    const f = vi.fn();
    vi.stubGlobal('fetch', f);
    const r = await embedTexts(settings, 'k', []);
    expect(r.vectors).toEqual([]);
    expect(f).not.toHaveBeenCalled();
  });

  it('无 apiKey 直接抛错', async () => {
    await expect(embedTexts(settings, '', ['a'])).rejects.toThrow(/API Key/);
  });

  it('按批量分批请求；服务端乱序返回时仍按 index 正确归位', async () => {
    const calls: unknown[] = [];
    vi.stubGlobal('fetch', async (_u: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { input: string[] };
      calls.push(body.input);
      // 语义正确但数组顺序被打乱的响应：第 0 项属于 input[1]，第 1 项属于 input[0]。
      // 生产代码必须用响应里的 index 归位，而不是数组下标，否则向量会与错文本绑定。
      const v = body.input.map((t) => (t === 'a' ? [1, 0] : t === 'b' ? [0, 1] : [2, 0]));
      const order = body.input.length === 2 ? [1, 0] : [0];
      const scrambled = [...v].reverse();
      return okResponse(scrambled, order) as unknown as Response;
    });
    const r = await embedTexts(settings, 'k', ['a', 'b', 'c']);
    expect(calls).toHaveLength(2); // batchSize=2
    expect(r.vectors).toEqual([
      [1, 0], // a
      [0, 1], // b
      [2, 0], // c
    ]);
    expect(r.dim).toBe(2);
  });

  it('返回条数不符时报错，不静默错位', async () => {
    vi.stubGlobal('fetch', async () => okResponse([[1, 0]]) as unknown as Response);
    await expect(embedTexts(settings, 'k', ['a', 'b'])).rejects.toThrow(/条数不符/);
  });

  it('401 立即失败、不重试', async () => {
    let n = 0;
    vi.stubGlobal('fetch', async () => {
      n++;
      return { ok: false, status: 401, text: async () => JSON.stringify({ error: { message: 'bad key' } }) } as unknown as Response;
    });
    await expect(embedTexts(settings, 'k', ['a'])).rejects.toThrow(/bad key/);
    expect(n).toBe(1);
  });

  it('dimensions 被 400 拒绝时自动去掉该参数重试一次', async () => {
    let n = 0;
    const bodies: Record<string, unknown>[] = [];
    vi.stubGlobal('fetch', async (_u: string, init: RequestInit) => {
      n++;
      const body = JSON.parse(String(init.body)) as Record<string, unknown>;
      bodies.push(body);
      if ('dimensions' in body) {
        return { ok: false, status: 400, text: async () => JSON.stringify({ error: { message: 'dimensions not supported' } }) } as unknown as Response;
      }
      return okResponse([[1, 0]]) as unknown as Response;
    });
    const r = await embedTexts(settings, 'k', ['a']);
    expect(n).toBe(2);
    expect(bodies[0].dimensions).toBe(1024);
    expect('dimensions' in bodies[1]).toBe(false);
    expect(r.dim).toBe(2);
  });

  it('429 会重试直到成功', async () => {
    let n = 0;
    vi.stubGlobal('fetch', async () => {
      n++;
      if (n < 3) return { ok: false, status: 429, text: async () => '{"error":{"message":"rate limited"}}' } as unknown as Response;
      return okResponse([[1, 0]]) as unknown as Response;
    });
    const r = await embedTexts(settings, 'k', ['a']);
    expect(n).toBe(3);
    expect(r.vectors).toEqual([[1, 0]]);
  });

  it('onProgress 报告进度', async () => {
    vi.stubGlobal('fetch', async (_u: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { input: string[] };
      return okResponse(body.input.map(() => [1, 0])) as unknown as Response;
    });
    const seen: number[] = [];
    await embedTexts(settings, 'k', ['a', 'b', 'c'], { onProgress: (done) => seen.push(done) });
    expect(seen).toEqual([2, 3]);
  });
});

describe('testEmbedding', () => {
  it('成功时返回维度与中文说明', async () => {
    vi.stubGlobal('fetch', async () => okResponse([[1, 2, 3]]) as unknown as Response);
    const r = await testEmbedding(settings, 'k');
    expect(r.ok).toBe(true);
    expect(r.dim).toBe(3);
    expect(r.messageZh).toContain('连接成功');
  });
  it('失败时返回 ok:false 与可读中文原因', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new Error('ECONNREFUSED');
    });
    const r = await testEmbedding(settings, 'k');
    expect(r.ok).toBe(false);
    expect(r.messageZh).toContain('连接失败');
  });
});
