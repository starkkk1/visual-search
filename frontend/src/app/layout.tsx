import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
});

export const metadata: Metadata = {
  title: "Visual Search | AI",
  description: "Find similar images instantly using deep learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${outfit.variable} dark antialiased h-full`}>
      <body className="min-h-full bg-zinc-950 text-zinc-50 font-outfit overflow-x-hidden selection:bg-purple-500/30">
        <div className="fixed top-[-20vh] left-[-20vw] w-[60vw] h-[60vw] bg-[radial-gradient(circle,rgba(139,92,246,0.15)_0%,rgba(0,0,0,0)_70%)] rounded-full -z-10 pointer-events-none" />
        {children}
      </body>
    </html>
  );
}
