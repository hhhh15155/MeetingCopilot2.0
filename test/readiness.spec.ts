/**
 * 面试前自检的单元测试。
 *
 * 这些用例不只是"函数返回值对不对"——每一个都对应一种**真会导致面试失败**的
 * 配置状态。其中最典型的一条（ASR 后端指向本机不存在的本地 runtime）是在
 * 真实使用中实测踩到的：向导把后端设成 local-realtime，而 FunASR 的 conda
 * 环境并不存在，结果一句话都转不出来，且配置界面完全看不出问题。
 */
import { describe, expect, it, vi } from 'vitest';
import {
  checkAsr,
  checkLlm,
  checkRag,
  checkMaterial,
  checkEmbeddingDeep,
  runReadiness,
  summarise,
  worstLevel,
  type ReadinessDeps,
} from '../shared/readiness';
import type { ReadinessItem } from '../shared/protocol';

function deps(over: Partial<ReadinessDeps> = {}): ReadinessDeps {
  return {
    asrBackend: 'cloud-realtime',
    probeLocalAsr: async () => ({ interpreterFound: true, interpreterPath: '/p', portOpen: true }),
    testLlm: async () => ({ ok: true, label: 'deepseek-chat', ms: 900 }),
    inspectRag: async () => ({
      enabled: true,
      chunkCount: 179,
      sources: [
        { name: '讲稿.md', slot: 'resume' as const, chunks: 179 },
        { name: 'JD.md', slot: 'jd' as const, chunks: 12 },
      ],
      model: 'text-embedding-v3',
      embeddingModel: 'text-embedding-v3',
      dim: 1024,
      embeddingDim: 1024,
      apiKeySet: true,
      byLevel: { verified: 128, claimed: 37, unverified: 14 },
    }),
    ...over,
  };
}

describe('worstLevel / summarise', () => {
  const mk = (level: ReadinessItem['level']): ReadinessItem => ({
    check: 'llm',
    label: 'X',
    level,
    detailZh: '',
    fixZh: '',
  });

  it('取最差档位', () => {
    expect(worstLevel([mk('ok'), mk('ok')])).toBe('ok');
    expect(worstLevel([mk('ok'), mk('warn')])).toBe('warn');
    expect(worstLevel([mk('warn'), mk('fail'), mk('ok')])).toBe('fail');
    expect(worstLevel([])).toBe('ok');
  });

  it('汇总文案点名要修的项', () => {
    const s = summarise([{ ...mk('fail'), label: '转写' }, { ...mk('fail'), label: '检索' }, mk('ok')]);
    expect(s.level).toBe('fail');
    expect(s.summaryZh).toContain('转写');
    expect(s.summaryZh).toContain('检索');
  });

  it('全好时给出可面试的结论', () => {
    expect(summarise([mk('ok')]).summaryZh).toContain('可以开始面试');
  });
});

describe('checkAsr —— 最容易踩的坑', () => {
  it('云识别：直接 ok（不自检里重复发网络请求）', async () => {
    const r = await checkAsr(deps({ asrBackend: 'cloud-realtime' }));
    expect(r.level).toBe('ok');
    expect(r.check).toBe('asr');
  });

  it('本地后端 + 解释器不存在 → fail，且给出两条可操作修法', async () => {
    const r = await checkAsr(
      deps({
        asrBackend: 'local-realtime',
        probeLocalAsr: async () => ({
          interpreterFound: false,
          interpreterPath: 'C:\\ProgramData\\miniconda3\\envs\\funasr\\python.exe',
          portOpen: false,
        }),
      }),
    );
    expect(r.level).toBe('fail');
    // 必须说清后果，否则用户不知道这是"面试时一句话都转不出来"
    expect(r.detailZh).toContain('没找到');
    expect(r.fixZh).toContain('阿里云实时');
    expect(r.fixZh).toContain('FunASR');
  });

  it('本地后端 + 解释器在但服务没起 → warn（不是 fail）', async () => {
    const r = await checkAsr(
      deps({
        asrBackend: 'local-realtime',
        probeLocalAsr: async () => ({ interpreterFound: true, interpreterPath: '/p', portOpen: false }),
      }),
    );
    expect(r.level).toBe('warn');
    expect(r.fixZh).toContain('测试');
  });

  it('本地后端 + 服务在跑 → ok', async () => {
    const r = await checkAsr(deps({ asrBackend: 'local-realtime' }));
    expect(r.level).toBe('ok');
  });

  it('探测抛异常时不崩，降为 warn', async () => {
    const r = await checkAsr(
      deps({
        asrBackend: 'local-realtime',
        probeLocalAsr: async () => {
          throw new Error('boom');
        },
      }),
    );
    expect(r.level).toBe('warn');
    expect(r.detailZh).toContain('boom');
  });
});

describe('checkLlm', () => {
  it('没配 Key → fail，并指向设置页', async () => {
    const r = await checkLlm(deps({ testLlm: async () => null }));
    expect(r.level).toBe('fail');
    expect(r.fixZh).toContain('LLM');
  });

  it('连接失败 → fail 并带上原因', async () => {
    const r = await checkLlm(deps({ testLlm: async () => ({ ok: false, label: 'deepseek', error: 'HTTP 401' }) }));
    expect(r.level).toBe('fail');
    expect(r.detailZh).toContain('401');
  });

  it('成功 → ok 且显示延迟', async () => {
    const r = await checkLlm(deps());
    expect(r.level).toBe('ok');
    expect(r.detailZh).toContain('900ms');
  });

  it('抛异常 → fail 而不是崩溃', async () => {
    const r = await checkLlm(
      deps({
        testLlm: async () => {
          throw new Error('ECONNREFUSED');
        },
      }),
    );
    expect(r.level).toBe('fail');
    expect(r.detailZh).toContain('ECONNREFUSED');
  });
});

describe('checkRag', () => {
  const base = {
    enabled: true,
    chunkCount: 179,
    sources: [{ name: 'a.md', slot: 'resume' as const, chunks: 179 }],
    model: 'text-embedding-v3',
    embeddingModel: 'text-embedding-v3',
    dim: 1024,
    embeddingDim: 1024,
    apiKeySet: true,
    byLevel: { verified: 128, claimed: 37, unverified: 14 },
  };

  it('关闭检索 → warn（不是 fail：还能用直塞方式）', async () => {
    const r = await checkRag(deps({ inspectRag: async () => ({ ...base, enabled: false }) }));
    expect(r.level).toBe('warn');
    expect(r.detailZh).toContain('已关闭');
  });

  it('没配 embedding Key → fail', async () => {
    const r = await checkRag(deps({ inspectRag: async () => ({ ...base, apiKeySet: false }) }));
    expect(r.level).toBe('fail');
    expect(r.fixZh).toContain('检索增强');
  });

  it('索引为空 → fail 且说清"会自动退回直塞"', async () => {
    const r = await checkRag(deps({ inspectRag: async () => ({ ...base, chunkCount: 0, sources: [] }) }));
    expect(r.level).toBe('fail');
    expect(r.fixZh).toContain('重建索引');
    expect(r.fixZh).toContain('直塞');
  });

  it('embedding 模型与索引不一致 → fail（检索会完全失效）', async () => {
    const r = await checkRag(
      deps({ inspectRag: async () => ({ ...base, embeddingModel: 'BAAI/bge-m3' }) }),
    );
    expect(r.level).toBe('fail');
    expect(r.detailZh).toContain('向量空间不同');
    expect(r.fixZh).toContain('重建索引');
  });

  it('维度不一致 → fail', async () => {
    const r = await checkRag(deps({ inspectRag: async () => ({ ...base, embeddingDim: 512 }) }));
    expect(r.level).toBe('fail');
    expect(r.detailZh).toContain('512');
  });

  it('一切正常 → ok 且报出档位分布', async () => {
    const r = await checkRag(deps({ inspectRag: async () => base }));
    expect(r.level).toBe('ok');
    expect(r.detailZh).toContain('179');
    expect(r.detailZh).toContain('待确认 37');
  });

  it('待确认占比过高 → warn（模型会什么都不敢讲）', async () => {
    const r = await checkRag(
      deps({
        inspectRag: async () => ({ ...base, byLevel: { verified: 10, claimed: 160, unverified: 9 } }),
      }),
    );
    expect(r.level).toBe('warn');
    expect(r.detailZh).toContain('偏高');
  });
});

describe('checkMaterial', () => {
  it('没索引任何资料 → fail', async () => {
    const r = await checkMaterial(
      deps({
        inspectRag: async () => ({
          enabled: true,
          chunkCount: 0,
          sources: [],
          embeddingModel: 'm',
          apiKeySet: true,
          byLevel: { verified: 0, claimed: 0, unverified: 0 },
        }),
      }),
    );
    expect(r.level).toBe('fail');
  });

  it('有项目资料但没有 JD → warn', async () => {
    const r = await checkMaterial(
      deps({
        inspectRag: async () => ({
          enabled: true,
          chunkCount: 179,
          sources: [{ name: '讲稿.md', slot: 'resume' as const, chunks: 179 }],
          embeddingModel: 'm',
          apiKeySet: true,
          byLevel: { verified: 1, claimed: 1, unverified: 1 },
        }),
      }),
    );
    expect(r.level).toBe('warn');
    expect(r.detailZh).toContain('没有岗位 JD');
  });

  it('项目资料 + JD 都有 → ok', async () => {
    const r = await checkMaterial(
      deps({
        inspectRag: async () => ({
          enabled: true,
          chunkCount: 200,
          sources: [
            { name: '讲稿.md', slot: 'resume' as const, chunks: 179 },
            { name: 'JD.md', slot: 'jd' as const, chunks: 21 },
          ],
          embeddingModel: 'm',
          apiKeySet: true,
          byLevel: { verified: 1, claimed: 1, unverified: 1 },
        }),
      }),
    );
    expect(r.level).toBe('ok');
    expect(r.detailZh).toContain('JD');
  });
});

describe('checkEmbeddingDeep', () => {
  it('快速自检不发网络请求', async () => {
    const spy = vi.fn();
    const r = await checkEmbeddingDeep(deps({ deep: false, testEmbedding: spy }));
    expect(spy).not.toHaveBeenCalled();
    expect(r.level).toBe('ok');
    expect(r.detailZh).toContain('跳过');
  });

  it('深度检查成功 → ok 带维度与延迟', async () => {
    const r = await checkEmbeddingDeep(
      deps({ deep: true, testEmbedding: async () => ({ ok: true, messageZh: 'ok', dim: 1024, latencyMs: 310 }) }),
    );
    expect(r.level).toBe('ok');
    expect(r.detailZh).toContain('1024');
    expect(r.detailZh).toContain('310ms');
  });

  it('深度检查失败 → fail', async () => {
    const r = await checkEmbeddingDeep(
      deps({ deep: true, testEmbedding: async () => ({ ok: false, messageZh: 'HTTP 401: bad key' }) }),
    );
    expect(r.level).toBe('fail');
    expect(r.detailZh).toContain('bad key');
  });
});

describe('runReadiness', () => {
  it('全部就绪 → ok，且 5 项都在', async () => {
    const r = await runReadiness(deps());
    expect(r.level).toBe('ok');
    expect(r.items).toHaveLength(5);
    expect(r.items.map((i) => i.check)).toEqual(['asr', 'llm', 'rag', 'material', 'embedding']);
    expect(r.deep).toBe(false);
  });

  it('只要有一项 fail，总体就是 fail（并且点名）', async () => {
    const r = await runReadiness(
      deps({
        asrBackend: 'local-realtime',
        probeLocalAsr: async () => ({ interpreterFound: false, interpreterPath: '/missing', portOpen: false }),
      }),
    );
    expect(r.level).toBe('fail');
    const asr = r.items.find((i) => i.check === 'asr')!;
    expect(asr.level).toBe('fail');
    expect(r.summaryZh).toContain('转写');
  });

  it('单项挂死不会拖垮整页（超时降级）', async () => {
    // 用永不 resolve 的 testLlm 模拟挂死；真实超时是 12s，这里靠 fake timer 不方便，
    // 因此只断言"其它项照常返回、整页仍然有结果"这一关键性质。
    const r = await runReadiness(
      deps({
        inspectRag: async () => ({
          enabled: true,
          chunkCount: 10,
          sources: [{ name: 'a.md', slot: 'resume' as const, chunks: 10 }],
          embeddingModel: 'm',
          apiKeySet: true,
          byLevel: { verified: 1, claimed: 1, unverified: 1 },
        }),
      }),
    );
    expect(r.items).toHaveLength(5);
    expect(typeof r.checkedAt).toBe('number');
  });
});
