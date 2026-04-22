"use client";

import React, { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from "recharts";
import {
  Wrench, ArrowRight, Loader2, CheckCircle2, XCircle,
  AlertTriangle, Download, Shield, ChevronDown, Code2,
  TrendingUp, Activity,
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ───────────────────────────────────────────────────────────────────

interface Improvement {
  metric_name: string;
  original_value: number;
  mitigated_value: number;
  threshold: number;
  original_passed: boolean;
  mitigated_passed: boolean;
  original_severity: string;
  mitigated_severity: string;
}

interface RemediationResult {
  session_id: string;
  remediation_id: string;
  strategy: string;
  model_type: string;
  original_accuracy: number;
  mitigated_accuracy: number;
  original_dir: number;
  mitigated_dir: number;
  improvements: Improvement[];
  script_diff: string;
  all_passed: boolean;
}

type Phase = "config" | "running" | "done" | "error";

const STRATEGIES = [
  {
    id: "reweighing",
    name: "Reweighing",
    desc: "Compute sample weights inversely proportional to group × label frequency, then retrain.",
    icon: "⚖️",
  },
  {
    id: "threshold_adjustment",
    name: "Threshold Adjustment",
    desc: "Find per-group classification thresholds that equalise selection rates (post-processing).",
    icon: "🎚️",
  },
  {
    id: "fairness_constraint",
    name: "Fairness Constraint",
    desc: "Use Fairlearn ExponentiatedGradient with DemographicParity constraint (in-processing).",
    icon: "🛡️",
  },
];

const METRIC_LABELS: Record<string, string> = {
  disparate_impact_ratio: "Disparate Impact Ratio",
  demographic_parity_difference: "Demographic Parity Diff",
  equalized_odds_difference: "Equalized Odds Diff",
};

const SEV_COLORS: Record<string, string> = {
  critical: "#ef4444",
  warning: "#f59e0b",
  pass: "#10b981",
};

// ═════════════════════════════════════════════════════════════════════════════
//  Page
// ═════════════════════════════════════════════════════════════════════════════

export default function RemediationPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  const [phase, setPhase] = useState<Phase>("config");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RemediationResult | null>(null);

  const [targetCol, setTargetCol] = useState("");
  const [sensitiveAttrs, setSensitiveAttrs] = useState("");
  const [strategy, setStrategy] = useState("reweighing");
  const [showDiff, setShowDiff] = useState(false);

  // ── Run remediation ──
  const runRemediation = useCallback(async () => {
    if (!targetCol.trim() || !sensitiveAttrs.trim()) {
      setError("Target column and sensitive attributes are required.");
      return;
    }
    setPhase("running");
    setError(null);

    try {
      const res = await fetch(`${API}/api/remediation/run/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: targetCol.trim(),
          sensitive_attributes: sensitiveAttrs.split(",").map((s) => s.trim()).filter(Boolean),
          strategy,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      }
      const data: RemediationResult = await res.json();
      setResult(data);
      setPhase("done");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Remediation failed");
      setPhase("error");
    }
  }, [sessionId, targetCol, sensitiveAttrs, strategy]);

  // ═════════════════════════════════════════════════════════════════════════
  //  Config Phase
  // ═════════════════════════════════════════════════════════════════════════

  if (phase === "config" || phase === "error") {
    return (
      <div className="min-h-screen bg-[#0a0a0f] text-gray-100">
        <div className="fixed inset-0 pointer-events-none">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-to-b from-emerald-500/[0.03] via-transparent to-transparent rounded-full blur-3xl" />
        </div>
        <div className="relative max-w-xl mx-auto px-4 py-16">
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/15 mb-4">
              <Wrench className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[11px] uppercase tracking-widest font-semibold text-emerald-400">Remediation</span>
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Mitigate Bias</h1>
            <p className="text-sm text-gray-500">Choose a strategy and retrain your model with fairness constraints.</p>
          </div>

          <div className="space-y-5 rounded-2xl border border-gray-700/40 bg-gray-900/50 p-6">
            {/* Target column */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1.5">Target Column <span className="text-red-400">*</span></label>
              <input id="rem-target-col" type="text" value={targetCol} onChange={(e) => setTargetCol(e.target.value)}
                placeholder="e.g. two_year_recid"
                className="w-full px-4 py-2.5 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-emerald-500/60 text-sm" />
            </div>

            {/* Sensitive attrs */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1.5">Protected Attributes <span className="text-red-400">*</span></label>
              <input id="rem-sensitive-attrs" type="text" value={sensitiveAttrs} onChange={(e) => setSensitiveAttrs(e.target.value)}
                placeholder="e.g. race, sex"
                className="w-full px-4 py-2.5 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-emerald-500/60 text-sm" />
            </div>

            {/* Strategy picker */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Mitigation Strategy</label>
              <div className="space-y-2">
                {STRATEGIES.map((s) => (
                  <button key={s.id} onClick={() => setStrategy(s.id)}
                    className={`w-full text-left flex items-start gap-3 p-3 rounded-xl border transition-all ${
                      strategy === s.id
                        ? "border-emerald-500/40 bg-emerald-500/10"
                        : "border-gray-700/40 bg-gray-800/30 hover:border-gray-600/50"
                    }`}>
                    <span className="text-xl mt-0.5">{s.icon}</span>
                    <div>
                      <p className={`text-sm font-semibold ${strategy === s.id ? "text-emerald-300" : "text-gray-200"}`}>{s.name}</p>
                      <p className="text-[11px] text-gray-500 mt-0.5">{s.desc}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {error && (
              <div className="flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3">
                <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <p className="text-xs text-red-300">{error}</p>
              </div>
            )}

            <button id="run-remediation-btn" onClick={runRemediation}
              className="w-full flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-semibold text-sm shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/40 hover:scale-[1.01] transition-all">
              <Wrench className="w-4 h-4" /> Run Remediation
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Running
  // ═════════════════════════════════════════════════════════════════════════

  if (phase === "running") {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 text-emerald-400 animate-spin mx-auto" />
          <h2 className="text-xl font-semibold text-white">Applying {STRATEGIES.find((s) => s.id === strategy)?.name}…</h2>
          <p className="text-sm text-gray-500 max-w-sm">Retraining the model with fairness constraints and computing new metrics.</p>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  Results
  // ═════════════════════════════════════════════════════════════════════════

  if (!result) return null;

  // Chart data for before/after comparison
  const comparisonData = result.improvements.map((imp) => ({
    name: (METRIC_LABELS[imp.metric_name] || imp.metric_name).replace(/ /g, "\n"),
    Original: imp.original_value,
    Mitigated: imp.mitigated_value,
    threshold: imp.threshold,
  }));

  const accDelta = result.mitigated_accuracy - result.original_accuracy;
  const dirDelta = result.mitigated_dir - result.original_dir;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 pb-20">
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-gradient-to-b from-emerald-500/[0.02] via-transparent to-transparent rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-5xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/15 mb-3">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[11px] uppercase tracking-widest font-semibold text-emerald-400">Remediation Complete</span>
            </div>
            <h1 className="text-2xl font-bold text-white mb-1">Remediation Report</h1>
            <p className="text-sm text-gray-500">
              Strategy: <span className="text-emerald-400 font-medium">{STRATEGIES.find((s) => s.id === result.strategy)?.name}</span>
              {" · "}{result.model_type}
            </p>
          </div>
          <div className={`px-5 py-3 rounded-xl border ${result.all_passed ? "border-emerald-500/30 bg-emerald-500/10" : "border-amber-500/30 bg-amber-500/10"}`}>
            <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Status</p>
            <p className={`text-lg font-black ${result.all_passed ? "text-emerald-400" : "text-amber-400"}`}>
              {result.all_passed ? "ALL PASSED" : "PARTIAL"}
            </p>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="rounded-xl bg-gray-900/60 border border-gray-700/30 px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-[11px] text-gray-500 uppercase tracking-wider">Original Acc</span>
            </div>
            <p className="text-lg font-bold text-white">{(result.original_accuracy * 100).toFixed(1)}%</p>
          </div>
          <div className="rounded-xl bg-gray-900/60 border border-gray-700/30 px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-[11px] text-gray-500 uppercase tracking-wider">Mitigated Acc</span>
            </div>
            <p className="text-lg font-bold text-white">
              {(result.mitigated_accuracy * 100).toFixed(1)}%
              <span className={`text-xs ml-1 ${accDelta >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                ({accDelta >= 0 ? "+" : ""}{(accDelta * 100).toFixed(1)}%)
              </span>
            </p>
          </div>
          <div className="rounded-xl bg-gray-900/60 border border-gray-700/30 px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-[11px] text-gray-500 uppercase tracking-wider">Original DIR</span>
            </div>
            <p className="text-lg font-bold text-white">{result.original_dir.toFixed(4)}</p>
          </div>
          <div className="rounded-xl bg-gray-900/60 border border-gray-700/30 px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-[11px] text-gray-500 uppercase tracking-wider">Mitigated DIR</span>
            </div>
            <p className="text-lg font-bold text-white">
              {result.mitigated_dir.toFixed(4)}
              <span className={`text-xs ml-1 ${dirDelta >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                ({dirDelta >= 0 ? "+" : ""}{dirDelta.toFixed(4)})
              </span>
            </p>
          </div>
        </div>

        {/* Before / After chart */}
        <div className="rounded-xl border border-gray-700/30 bg-gray-900/50 p-5 mb-8">
          <h3 className="text-sm font-semibold text-white mb-4">Before vs After — Fairness Metrics</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} />
                <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Original" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Mitigated" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Per-metric improvement table */}
        <div className="rounded-xl border border-gray-700/30 bg-gray-900/50 p-5 mb-8">
          <h3 className="text-sm font-semibold text-white mb-4">Metric Improvements</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-700/40">
                  <th className="text-left py-2 pr-4">Metric</th>
                  <th className="text-right py-2 px-3">Original</th>
                  <th className="text-right py-2 px-3">Mitigated</th>
                  <th className="text-right py-2 px-3">Threshold</th>
                  <th className="text-center py-2 pl-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {result.improvements.map((imp) => (
                  <tr key={imp.metric_name} className="border-b border-gray-800/40">
                    <td className="py-2 pr-4 text-gray-300 font-medium">{METRIC_LABELS[imp.metric_name] || imp.metric_name}</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: SEV_COLORS[imp.original_severity] || "#94a3b8" }}>
                      {imp.original_value.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: SEV_COLORS[imp.mitigated_severity] || "#94a3b8" }}>
                      {imp.mitigated_value.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-500">{imp.threshold}</td>
                    <td className="py-2 pl-3 text-center">
                      {imp.mitigated_passed
                        ? <CheckCircle2 className="w-4 h-4 text-emerald-400 inline" />
                        : <AlertTriangle className="w-4 h-4 text-amber-400 inline" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Script diff */}
        {result.script_diff && (
          <div className="rounded-xl border border-gray-700/30 bg-gray-900/50 mb-8 overflow-hidden">
            <button onClick={() => setShowDiff(!showDiff)}
              className="w-full flex items-center justify-between px-5 py-3 text-left hover:bg-gray-800/30 transition-colors">
              <span className="flex items-center gap-2 text-sm font-semibold text-gray-200">
                <Code2 className="w-4 h-4 text-blue-400" /> Script Diff
              </span>
              <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showDiff ? "rotate-180" : ""}`} />
            </button>
            {showDiff && (
              <div className="px-5 pb-5 border-t border-gray-700/30">
                <pre className="text-xs font-mono leading-relaxed overflow-x-auto mt-3 max-h-96 overflow-y-auto">
                  {result.script_diff.split("\n").map((line, i) => {
                    let color = "text-gray-500";
                    if (line.startsWith("+") && !line.startsWith("+++")) color = "text-emerald-400";
                    else if (line.startsWith("-") && !line.startsWith("---")) color = "text-red-400";
                    else if (line.startsWith("@@")) color = "text-blue-400";
                    return <div key={i} className={color}>{line}</div>;
                  })}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between">
          <button onClick={() => window.open(`${API}/api/remediation/${sessionId}/download`, "_blank")}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium border border-gray-600/40 text-gray-300 hover:bg-gray-800/50 transition-all">
            <Download className="w-4 h-4" /> Download Mitigated Model
          </button>
          <button id="back-to-upload" onClick={() => router.push("/upload")}
            className="group flex items-center gap-2.5 px-6 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-amber-500 to-amber-600 text-gray-950 shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40 hover:scale-[1.02] transition-all">
            Start New Audit
            <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>
      </div>
    </div>
  );
}
