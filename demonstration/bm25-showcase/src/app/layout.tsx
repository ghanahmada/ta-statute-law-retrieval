import type { Metadata } from "next";
import { DM_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const dmSans = DM_Sans({ variable: "--font-display", subsets: ["latin"] });
const ibmMono = IBM_Plex_Mono({ variable: "--font-mono", weight: ["400", "500", "600"], subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Retrieval Showcase — KUHPerdata",
  description: "Legal statute retrieval analysis for expert review",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id" className={`${dmSans.variable} ${ibmMono.variable} h-full antialiased`}>
      <body className="min-h-full font-[family-name:var(--font-display)]">{children}</body>
    </html>
  );
}
