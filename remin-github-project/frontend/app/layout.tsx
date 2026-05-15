import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Remin 学术文献智能分析系统",
  description:
    "基于 RAG 与大模型微调技术的学术辅助平台，支持高价值论文筛选、证据溯源问答与自动生成综述报告。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
