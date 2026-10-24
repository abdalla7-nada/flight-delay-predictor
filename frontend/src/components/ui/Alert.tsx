import React from "react";
import { AlertTriangle, CheckCircle, Info, XCircle } from "lucide-react";

interface AlertProps {
  title?: string;
  children: React.ReactNode;
  variant?: "info" | "warning" | "error" | "success";
  className?: string;
}

export function Alert({ title, children, variant = "info", className = "" }: AlertProps) {
  const styles = {
    info: {
      container: "bg-blue-950/40 border-blue-800/60 text-blue-200",
      icon: <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />,
    },
    warning: {
      container: "bg-amber-950/40 border-amber-800/60 text-amber-200",
      icon: <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />,
    },
    error: {
      container: "bg-rose-950/40 border-rose-800/60 text-rose-200",
      icon: <XCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />,
    },
    success: {
      container: "bg-emerald-950/40 border-emerald-800/60 text-emerald-200",
      icon: <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />,
    },
  };

  const { container, icon } = styles[variant];

  return (
    <div className={`flex items-start gap-2.5 sm:gap-3 p-3 sm:p-4 rounded-xl border ${container} ${className}`} role="alert">
      {icon}
      <div className="text-xs sm:text-sm space-y-1 min-w-0 flex-1 break-words">
        {title && <h5 className="font-semibold text-white">{title}</h5>}
        <div className="text-slate-300 leading-relaxed">{children}</div>
      </div>
    </div>
  );
}
