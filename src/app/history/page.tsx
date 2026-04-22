"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  History, Loader2, ChevronRight, Scale, AlertTriangle, CheckCircle2,
  XCircle, FileText, Activity, ShieldAlert, Search
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SessionData {
  session_id: string;
  dataset_filename: string;
  model_filename: string | null;
  status: string;
  row_count: number | null;
  feature_count: number | null;
  created_at: string;
  metric_count: number;
  overall_severity: "none" | "pass" | "warning" | "critical";
  verdict: {
    verdict: "guilty" | "not_guilty";
    risk_score: number;
  } | null;
  remediation: {
    strategy: string;
    status: string;
    mitigated_dir: number;
  } | null;
}

export default function HistoryPage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchSessions() {
      try {
        const res = await fetch(`${API}/api/sessions`);
        if (!res.ok) throw new Error("Failed to load history");
        const data = await res.json();
        setSessions(data.sessions || []);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Error loading sessions");
      } finally {
        setLoading(false);
      }
    }
    fetchSessions();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 pb-20">
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-to-b from-blue-500/[0.03] via-transparent to-transparent rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-6xl mx-auto px-4 py-16">
        <div className="mb-10 flex items-end justify-between">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/15 mb-3">
              <History className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-[11px] uppercase tracking-widest font-semibold text-blue-400">Audit Logs</span>
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">Session History</h1>
            <p className="text-sm text-gray-500">View and resume past AI Courtroom bias audits.</p>
          </div>
          <button
            onClick={() => router.push("/upload")}
            className="px-5 py-2.5 bg-white text-gray-950 text-sm font-semibold rounded-lg hover:bg-gray-200 transition-colors shadow-sm"
          >
            Start New Audit
          </button>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl mb-8 flex items-center gap-2">
            <XCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        )}

        {sessions.length === 0 && !error ? (
          <div className="text-center py-24 border border-dashed border-gray-700/50 rounded-2xl bg-gray-900/30">
            <Activity className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-300 mb-1">No audits found</h3>
            <p className="text-sm text-gray-500">Upload a dataset and model to start your first bias audit.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {sessions.map((s) => {
              // Decide which phase they reached
              let targetRoute = `/upload`;
              let phaseName = "Upload";
              let progressColor = "text-gray-400";
              
              if (s.status === "complete") {
                targetRoute = `/analysis/${s.session_id}`;
                phaseName = "Analysis";
                progressColor = "text-blue-400";
              }
              if (s.verdict) {
                targetRoute = `/courtroom/${s.session_id}`;
                phaseName = "Courtroom";
                progressColor = "text-amber-400";
              }
              if (s.remediation) {
                targetRoute = `/remediation/${s.session_id}`;
                phaseName = "Remediated";
                progressColor = "text-emerald-400";
              }

              return (
                <div
                  key={s.session_id}
                  onClick={() => router.push(targetRoute)}
                  className="group flex flex-col md:flex-row md:items-center gap-6 p-5 rounded-2xl border border-gray-700/40 bg-gray-900/50 hover:bg-gray-800/60 hover:border-gray-600/60 transition-all cursor-pointer"
                >
                  {/* Left: Info */}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-gray-200 text-lg group-hover:text-white transition-colors">
                        {s.dataset_filename}
                      </h3>
                      <span className="text-xs font-mono text-gray-500 bg-gray-950/50 px-2 py-0.5 rounded border border-gray-800">
                        {s.session_id.split("-")[0]}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-gray-500">
                      <span>{new Date(s.created_at).toLocaleString()}</span>
                      {s.model_filename && (
                        <>
                          <span>•</span>
                          <span className="flex items-center gap-1.5"><FileText className="w-3.5 h-3.5" /> {s.model_filename}</span>
                        </>
                      )}
                      {s.row_count && (
                        <>
                          <span>•</span>
                          <span>{s.row_count.toLocaleString()} rows</span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Middle: Badges */}
                  <div className="flex items-center gap-3 md:w-auto w-full flex-wrap">
                    {/* Severity Badge */}
                    {s.metric_count > 0 && (
                      <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border
                        ${s.overall_severity === 'critical' ? 'bg-red-500/10 border-red-500/20 text-red-400' :
                          s.overall_severity === 'warning' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
                          'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'}`}>
                        {s.overall_severity === 'critical' ? <XCircle className="w-3.5 h-3.5" /> :
                         s.overall_severity === 'warning' ? <AlertTriangle className="w-3.5 h-3.5" /> :
                         <CheckCircle2 className="w-3.5 h-3.5" />}
                        {s.metric_count} Metrics
                      </div>
                    )}

                    {/* Verdict Badge */}
                    {s.verdict && (
                      <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border
                        ${s.verdict.verdict === 'guilty' ? 'bg-red-500/10 border-red-500/20 text-red-400' : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'}`}>
                        <Scale className="w-3.5 h-3.5" />
                        {s.verdict.verdict === 'guilty' ? 'GUILTY' : 'NOT GUILTY'}
                      </div>
                    )}

                    {/* Remediation Badge */}
                    {s.remediation && (
                      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-emerald-500/10 border-emerald-500/20 text-emerald-400">
                        <ShieldAlert className="w-3.5 h-3.5" />
                        Mitigated
                      </div>
                    )}
                  </div>

                  {/* Right: Quick Actions */}
                  <div className="flex flex-col md:flex-row items-center gap-3 md:w-auto">
                    <div className="flex items-center gap-2">
                      {s.verdict && (
                        <button 
                          onClick={(e) => { e.stopPropagation(); router.push(`/courtroom/${s.session_id}`); }}
                          className="px-3 py-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 text-[10px] font-bold rounded-lg border border-amber-500/20 transition-all uppercase tracking-wider"
                        >
                          Verdict
                        </button>
                      )}
                      {s.remediation && (
                        <button 
                          onClick={(e) => { e.stopPropagation(); router.push(`/remediation/${s.session_id}`); }}
                          className="px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 text-[10px] font-bold rounded-lg border border-emerald-500/20 transition-all uppercase tracking-wider"
                        >
                          Mitigation
                        </button>
                      )}
                      <button 
                        onClick={(e) => { e.stopPropagation(); router.push(`/explain/${s.session_id}`); }}
                        className="px-3 py-1.5 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 text-[10px] font-bold rounded-lg border border-blue-500/20 transition-all uppercase tracking-wider"
                      >
                        Explain
                      </button>
                    </div>

                    <div className="flex items-center gap-2 h-full border-l border-gray-700/50 pl-3 ml-1">
                      <button 
                        onClick={(e) => { e.stopPropagation(); window.open(`${API}/api/reports/${s.session_id}/pdf`); }}
                        className="p-2 text-gray-500 hover:text-red-400 transition-colors"
                        title="PDF Report"
                      >
                        <FileText className="w-4 h-4" />
                      </button>
                      <ChevronRight className="w-5 h-5 text-gray-600 group-hover:text-white transition-colors" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
