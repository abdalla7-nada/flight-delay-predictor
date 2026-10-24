import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#090d16",
};

export const metadata: Metadata = {
  title: "AeroRisk | Flight Delay Risk Prediction System",
  description:
    "Production-grade flight delay risk prediction platform powered by an audited CatBoost GBDT model with Platt scaling.",
  keywords: ["flight delay", "risk prediction", "machine learning", "catboost", "aviation analytics"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body
        suppressHydrationWarning
        className="min-h-full flex flex-col bg-[#090d16] text-slate-100 selection:bg-indigo-500/30 selection:text-white overflow-x-hidden w-full"
      >
        {children}
      </body>
    </html>
  );
}
