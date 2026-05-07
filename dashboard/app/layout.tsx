import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "genai-eval",
  description: "Multilingual GenAI evaluation dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        <div className="mx-auto max-w-6xl px-6 py-8">
          <header className="mb-8 border-b border-zinc-800 pb-4">
            <h1 className="text-2xl font-semibold tracking-tight">genai-eval</h1>
            <p className="mt-1 text-sm text-zinc-400">
              Pass rates and regression trends per model, task, and language
            </p>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
