"use client";

import React from "react";
import { Plane, Activity, FileSpreadsheet, Info } from "lucide-react";

interface NavbarProps {
  activeTab: "single" | "batch" | "audit" | "transparency";
  setActiveTab: (tab: "single" | "batch" | "audit") => void;
}

export function Navbar({ activeTab, setActiveTab }: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-2 sm:gap-4">
          {/* Brand */}
          <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white shrink-0">
              <Plane className="w-4 h-4 sm:w-5 sm:h-5 -rotate-45" />
            </div>
            <div>
              <span className="font-extrabold text-sm sm:text-base tracking-tight text-white block">
                AeroRisk
              </span>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Flight Delay Risk Classifier & Stratifier
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 sm:gap-2 overflow-x-auto no-scrollbar flex-nowrap shrink-0 py-1 max-w-[65%] sm:max-w-none">
            <button
              type="button"
              onClick={() => setActiveTab("single")}
              className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-4 py-2 min-h-[40px] sm:min-h-[44px] rounded-xl text-xs sm:text-sm font-semibold transition-all shrink-0 cursor-pointer ${
                activeTab === "single"
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-900/60"
              }`}
            >
              <Activity className="w-4 h-4 shrink-0" />
              <span className="whitespace-nowrap">
                <span className="inline sm:hidden">Single</span>
                <span className="hidden sm:inline">Single Flight</span>
              </span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("batch")}
              className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-4 py-2 min-h-[40px] sm:min-h-[44px] rounded-xl text-xs sm:text-sm font-semibold transition-all shrink-0 cursor-pointer ${
                activeTab === "batch"
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-900/60"
              }`}
            >
              <FileSpreadsheet className="w-4 h-4 shrink-0" />
              <span className="whitespace-nowrap">
                <span className="inline sm:hidden">Batch</span>
                <span className="hidden sm:inline">Batch Scoring</span>
              </span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("audit")}
              className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-4 py-2 min-h-[40px] sm:min-h-[44px] rounded-xl text-xs sm:text-sm font-semibold transition-all shrink-0 cursor-pointer ${
                activeTab === "transparency" || activeTab === "audit"
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-900/60"
              }`}
            >
              <Info className="w-4 h-4 shrink-0" />
              <span className="whitespace-nowrap">
                <span className="inline sm:hidden">Audit</span>
                <span className="hidden sm:inline">Model Audit & Transparency</span>
              </span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Navbar;
