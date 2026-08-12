import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PrepAgent",
  description: "Agentic RAG interview-prep copilot for DSA and SQL",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
