import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Scale } from "lucide-react";
import Link from "next/link";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "TrialAI",
  description: "Put your AI on trial before the world does.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased bg-background text-foreground flex flex-col min-h-screen`}
      >
        <nav className="sticky top-0 z-50 w-full border-b border-border bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 font-bold text-xl tracking-tight text-foreground">
              <Scale className="w-6 h-6 text-gold" />
              <span>TrialAI</span>
            </Link>
            <div className="flex items-center gap-6">
              <Link href="/demo" className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors">
                Run Demo
              </Link>
              <Link href="/trial/upload" className="text-sm font-medium bg-foreground text-background px-4 py-2 rounded-md hover:bg-foreground/90 transition-all shadow-sm">
                Start Trial
              </Link>
            </div>
          </div>
        </nav>
        <main className="flex-1 flex flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
