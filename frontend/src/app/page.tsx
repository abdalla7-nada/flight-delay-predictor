'use client';

import { useState, useSyncExternalStore } from 'react';
import Navbar from '@/components/Navbar';
import SinglePredictor from '@/components/SinglePredictor';
import BatchPredictor from '@/components/BatchPredictor';
import ModelTransparency from '@/components/ModelTransparency';

const emptySubscribe = () => () => {};

export default function Home() {
  const [activeTab, setActiveTab] = useState<'single' | 'batch' | 'audit'>('single');
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#090d16] text-slate-400">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="font-mono text-sm">Loading AeroRisk Engine...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#090d16] text-slate-100 overflow-x-hidden w-full">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 lg:px-8 py-4 sm:py-8">
        {activeTab === 'single' && <SinglePredictor />}
        {activeTab === 'batch' && <BatchPredictor />}
        {activeTab === 'audit' && <ModelTransparency />}
      </main>
      {/* Clean footer without the champion badge */}
      <footer className="border-t border-slate-800/80 bg-slate-950/40 py-4 sm:py-6 px-4 text-center text-[11px] sm:text-xs text-slate-500 font-mono">
        Schedule-Only Pre-Departure Flight Risk Engine &bull; Strictly Zero Leakage
      </footer>
    </div>
  );
}
