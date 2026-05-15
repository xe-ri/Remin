"use client";

import { ChangeEvent, FormEvent, useState } from "react";

type SearchResult = {
  paperId: string;
  title: string;
  abstract: string;
  year?: number;
  authors: string[];
  similarity_score: number;
  domain_score?: number;
  remin_score: number;
};

type UploadResult = {
  source_id: string;
  filename: string;
  title: string;
  chunk_count: number;
};

const coreValues = [
  "智能交通领域约束检索与推荐",
  "摘要相关性 + 领域相关性 + 时效性联合排序",
  "ChromaDB 证据检索约束生成过程",
  "支持文献分析问答与综述生成",
];

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

const REQUEST_TIMEOUT_MS = 60_000;

async function requestJson<T>(
  path: string,
  options: RequestInit,
  timeoutMessage: string,
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      ...options,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "";
      throw new Error(detail || `请求失败：HTTP ${response.status}`);
    }

    return data as T;
  } catch (requestError) {
    if (requestError instanceof DOMException && requestError.name === "AbortError") {
      throw new Error(`${timeoutMessage}。当前后端地址：${apiBaseUrl}`);
    }

    if (requestError instanceof TypeError) {
      throw new Error(
        `无法连接后端服务。请确认 ${apiBaseUrl} 可以在浏览器中打开，或检查前端 .env.local 配置。`,
      );
    }

    throw requestError;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export default function Home() {
  const [keyword, setKeyword] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [resultSource, setResultSource] = useState<"arxiv" | "fallback" | "">("");
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [uploadedSources, setUploadedSources] = useState<UploadResult[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);

  const togglePaper = (paperId: string) => {
    setSelectedPaperIds((current) =>
      current.includes(paperId)
        ? current.filter((id) => id !== paperId)
        : [...current, paperId],
    );
  };

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!keyword.trim()) {
      setError("请输入研究主题后再搜索。");
      return;
    }

    setSearchLoading(true);
    setError("");

    try {
      const data = await requestJson<{
        results?: SearchResult[];
        result_source?: "arxiv" | "fallback" | "";
      }>("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyword, limit: 8 }),
      }, "搜索请求超时，可能是服务器后端不可达或 arXiv 检索耗时过长");

      setResults(data.results || []);
      setResultSource(data.result_source || "");
      setSelectedPaperIds([]);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "搜索失败，请稍后重试。",
      );
    } finally {
      setSearchLoading(false);
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] ?? null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("请选择一个 PDF 文件后再上传。");
      return;
    }

    setUploadLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("title", selectedFile.name.replace(/\.pdf$/i, ""));

      const data = await requestJson<UploadResult>("/upload", {
        method: "POST",
        body: formData,
      }, "上传请求超时，可能是服务器后端不可达或 PDF 解析耗时过长");

      setUploadedSources((current) => [...current, data]);
      setSelectedFile(null);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "上传失败，请稍后重试。",
      );
    } finally {
      setUploadLoading(false);
    }
  };

  const handleAsk = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("请输入问题后再发起分析。");
      return;
    }

    if (selectedPaperIds.length === 0 && uploadedSources.length === 0) {
      setError("请先勾选推荐文献，或者上传 PDF 文件。");
      return;
    }

    setChatLoading(true);
    setError("");

    try {
      const data = await requestJson<{ answer?: string }>("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          paper_ids: selectedPaperIds,
          uploaded_source_ids: uploadedSources.map((item) => item.source_id),
        }),
      }, "分析请求超时，可能是服务器后端不可达或模型生成耗时过长");

      setAnswer(data.answer || "");
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "分析失败，请稍后重试。",
      );
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[var(--color-page)] text-[var(--color-ink)]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[-8rem] top-[-5rem] h-72 w-72 rounded-full bg-[radial-gradient(circle,_rgba(244,114,182,0.22),_transparent_70%)] blur-2xl" />
        <div className="absolute right-[-6rem] top-24 h-80 w-80 rounded-full bg-[radial-gradient(circle,_rgba(14,165,233,0.2),_transparent_72%)] blur-3xl" />
        <div className="absolute bottom-[-10rem] left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-[radial-gradient(circle,_rgba(250,204,21,0.16),_transparent_72%)] blur-3xl" />
        <div className="grid-overlay absolute inset-0 opacity-40" />
      </div>

      <section className="relative mx-auto max-w-7xl px-6 py-10 sm:px-10 lg:px-16">
        <header className="grid gap-6 border-b border-black/10 pb-8 lg:grid-cols-[1.4fr_0.8fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.28em] text-[var(--color-accent)]">
              Remin Intelligent Transportation Literature System
            </p>
            <h1 className="mt-4 max-w-4xl text-5xl font-semibold leading-tight sm:text-6xl">
              智能交通系统文献在线分析工具
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-black/72">
              输入智能交通系统相关研究主题后，系统会依据摘要与关键词相关性、领域相关性及时效性综合推荐论文；
              你也可以上传自己的 PDF，再让模型基于证据完成更学术、更严谨的文献分析问答与综述生成。
            </p>
          </div>

          <div className="glass-panel rounded-[2rem] p-6">
            <p className="text-sm uppercase tracking-[0.24em] text-black/55">
              当前能力
            </p>
            <div className="mt-4 space-y-3 text-sm leading-7 text-black/72">
              {coreValues.map((item) => (
                <div
                  key={item}
                  className="rounded-[1rem] border border-black/8 bg-white/70 px-4 py-3"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
        </header>

        <div className="mt-10 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="space-y-6">
            <form onSubmit={handleSearch} className="glass-panel rounded-[2rem] p-6 sm:p-8">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold">1. 主题搜索</h2>
                  <p className="mt-2 text-sm leading-7 text-black/68">
                    输入智能交通研究方向，系统将自动加入交通领域约束，并基于摘要相关性、领域相关性和发表时间返回推荐文献。
                  </p>
                </div>
                <span className="rounded-full bg-white/70 px-4 py-2 text-xs tracking-[0.18em] text-black/60">
                  Search
                </span>
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <input
                  value={keyword}
                  onChange={(event) => setKeyword(event.target.value)}
                  className="w-full rounded-[1rem] border border-black/10 bg-white/80 px-4 py-3 outline-none ring-0 placeholder:text-black/35"
                  placeholder="例如：traffic flow prediction, V2X communication, traffic signal control..."
                />
                <button
                  type="submit"
                  disabled={searchLoading}
                  className="rounded-[1rem] bg-[var(--color-ink)] px-6 py-3 text-sm font-semibold text-[var(--color-page)] transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {searchLoading ? "搜索中..." : "开始搜索"}
                </button>
              </div>

              <p className="mt-3 text-xs text-black/45">
                当前后端地址：{apiBaseUrl}
              </p>

              {error ? (
                <div className="mt-4 rounded-[1rem] border border-red-300/60 bg-red-50 px-4 py-3 text-sm leading-6 text-red-700">
                  {error}
                </div>
              ) : null}

              {resultSource ? (
                <div className="mt-4 rounded-[1rem] border border-black/8 bg-white/70 px-4 py-3 text-sm text-black/62">
                  {resultSource === "arxiv"
                    ? "当前结果来源：arXiv 在线预印本检索"
                    : "当前结果来源：离线演示数据（arXiv 接口暂不可用）"}
                </div>
              ) : null}
            </form>

            <div className="glass-panel rounded-[2rem] p-6 sm:p-8">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold">2. 推荐文献</h2>
                  <p className="mt-2 text-sm leading-7 text-black/68">
                    勾选要纳入分析的论文，系统会优先尝试下载全文；若无法获取全文，再回退到摘要参与问答。
                  </p>
                </div>
                <span className="rounded-full bg-white/70 px-4 py-2 text-xs tracking-[0.18em] text-black/60">
                  Ranking
                </span>
              </div>

              <div className="mt-6 space-y-4">
                {results.length === 0 ? (
                  <p className="rounded-[1.2rem] border border-dashed border-black/15 bg-white/55 px-4 py-6 text-sm text-black/55">
                    还没有搜索结果。先输入主题，系统会返回推荐论文列表。
                  </p>
                ) : (
                  results.map((paper) => {
                    const selected = selectedPaperIds.includes(paper.paperId);
                    return (
                      <label
                        key={paper.paperId}
                        className={`block cursor-pointer rounded-[1.4rem] border px-5 py-5 transition ${
                          selected
                            ? "border-[var(--color-accent)] bg-amber-50/80"
                            : "border-black/8 bg-white/72"
                        }`}
                      >
                        <div className="flex items-start gap-4">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => togglePaper(paper.paperId)}
                            className="mt-1 h-4 w-4 rounded border-black/20"
                          />
                          <div className="flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="text-lg font-semibold">{paper.title}</h3>
                              {paper.year ? (
                                <span className="text-sm text-black/45">({paper.year})</span>
                              ) : null}
                            </div>
                            <p className="mt-2 text-sm text-black/55">
                              {paper.authors.join(", ") || "作者信息缺失"}
                            </p>
                            <p className="mt-3 text-sm leading-7 text-black/70">
                              {paper.abstract}
                            </p>
                            <div className="mt-4 flex flex-wrap gap-2 text-xs text-black/60">
                              <span className="rounded-full bg-white/80 px-3 py-2">
                                综合评分 {paper.remin_score}
                              </span>
                              <span className="rounded-full bg-white/80 px-3 py-2">
                                相关度 {paper.similarity_score}
                              </span>
                              <span className="rounded-full bg-white/80 px-3 py-2">
                                领域分 {paper.domain_score ?? 0}
                              </span>
                            </div>
                          </div>
                        </div>
                      </label>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          <aside className="space-y-6">
            <section className="glass-panel rounded-[2rem] p-6 sm:p-8">
              <h2 className="text-2xl font-semibold">3. 上传 PDF</h2>
              <p className="mt-2 text-sm leading-7 text-black/68">
                如果你有自己的论文全文，也可以直接上传，系统会切片后写入 ChromaDB。
              </p>

              <div className="mt-6 space-y-3">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="block w-full rounded-[1rem] border border-black/10 bg-white/80 px-4 py-3 text-sm"
                />
                <button
                  type="button"
                  onClick={handleUpload}
                  disabled={uploadLoading}
                  className="w-full rounded-[1rem] border border-black/10 bg-white/85 px-4 py-3 text-sm font-semibold transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {uploadLoading ? "上传中..." : "上传并入库"}
                </button>
              </div>

              <div className="mt-5 space-y-3">
                {uploadedSources.length === 0 ? (
                  <p className="text-sm text-black/50">还没有上传文件。</p>
                ) : (
                  uploadedSources.map((item) => (
                    <div
                      key={item.source_id}
                      className="rounded-[1rem] border border-black/8 bg-white/70 px-4 py-3"
                    >
                      <p className="font-medium">{item.title}</p>
                      <p className="mt-1 text-xs text-black/55">
                        source_id: {item.source_id}
                      </p>
                      <p className="mt-1 text-xs text-black/55">
                        已切片 {item.chunk_count} 段
                      </p>
                    </div>
                  ))
                )}
              </div>
            </section>

            <form onSubmit={handleAsk} className="glass-panel rounded-[2rem] p-6 sm:p-8">
              <h2 className="text-2xl font-semibold">4. 学术分析问答</h2>
              <p className="mt-2 text-sm leading-7 text-black/68">
                系统会优先依据你勾选的推荐文献和已上传 PDF 进行证据检索，再围绕你的具体问题生成更审慎的文献分析回答，也可用于综述生成。
              </p>

              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="mt-6 min-h-40 w-full rounded-[1.2rem] border border-black/10 bg-white/80 px-4 py-4 text-sm outline-none placeholder:text-black/35"
                placeholder="例如：这些文献分别解决了什么问题？或请比较它们在交通流预测中的主要方法、优势与局限。"
              />

              <button
                type="submit"
                disabled={chatLoading}
                className="mt-4 w-full rounded-[1rem] bg-[var(--color-accent)] px-4 py-3 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {chatLoading ? "分析中..." : "生成学术性回答"}
              </button>

              <div className="mt-6 rounded-[1.2rem] border border-black/8 bg-white/70 p-4">
                <p className="text-xs uppercase tracking-[0.22em] text-black/45">Answer</p>
                <div className="mt-3 whitespace-pre-wrap text-sm leading-7 text-black/74">
                  {answer || "这里会显示基于文献证据生成的回答。"}
                </div>
              </div>
            </form>
          </aside>
        </div>

        {error ? (
          <div className="mt-6 rounded-[1.2rem] border border-red-300/60 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        ) : null}
      </section>
    </main>
  );
}
