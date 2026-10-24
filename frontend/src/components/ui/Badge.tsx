import React from "react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "low" | "below" | "average" | "above" | "high" | "neutral" | "info" | "success" | "warning";
  className?: string;
}

export function Badge({ children, variant = "neutral", className = "" }: BadgeProps) {
  const variantStyles = {
    low: "bg-emerald-950/70 text-emerald-400 border-emerald-800/60",
    below: "bg-sky-950/70 text-sky-400 border-sky-800/60",
    average: "bg-amber-950/70 text-amber-400 border-amber-800/60",
    above: "bg-orange-950/70 text-orange-400 border-orange-800/60",
    high: "bg-rose-950/70 text-rose-400 border-rose-800/60",
    neutral: "bg-slate-800/70 text-slate-300 border-slate-700/60",
    info: "bg-blue-950/70 text-blue-400 border-blue-800/60",
    success: "bg-emerald-950/70 text-emerald-400 border-emerald-800/60",
    warning: "bg-amber-950/70 text-amber-400 border-amber-800/60",
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
