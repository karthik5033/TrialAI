"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from "recharts";
import { 
  Search, BrainCircuit, ShieldAlert, BookOpen, Loader2, Play, Users, Briefcase, Gavel, XCircle, ChevronRight, CheckCircle2, AlertTriangle
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ExplainMode = "technical" | "manager" | "legal";

interface NarrativeResponse {
  mode: string;
  headline: string;
  verdict: string;
  why_summary: string;
  top_factors: Array<{
    feature: string;
    original_value: string;
    direction: string;
    relative_importance: string;
    commentary: string;
  }>;
  counterfactual_insights: {
    summary: string;
    examples: Array<{
      description: string;
      changed_features: Record<string, string>;
      new_prediction: string;
    }>;
  };
  fairness_and_risk: {
    uses_protected_attributes: string;
    protected_attribute_impact: string;
    notes: string;
  };
  recommended_actions: string[];
  error?: string;
  raw_text?: string;
}

export default function ExplainabilityPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;

  const [rowIndex, setRowIndex] = useState(0);
  const [targetCol, setTargetCol] = useState("");
  const [sensitiveAttrs, setSensitiveAttrs] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [shapData, setShapData] = useState<any>(null);
  const [cfData, setCfData] = useState<any>(null);
  const [narratives, setNarratives] = useState<Record<string, NarrativeResponse>>({});
  const [loadingNarrative, setLoadingNarrative] = useState(false);
  const [currentMode, setCurrentMode] = useState<ExplainMode>("manager");

  // Run all
  const analyzeRow = async () => {
    if (!targetCol || !sensitiveAttrs) {
      setError("Please provide Target Column and Protected Attributes.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setShapData(null);
    setCfData(null);
    setNarratives({});

    try {
      const payload = {
        row_index: rowIndex,
        target_column: targetCol,
        sensitive_attributes: sensitiveAttrs.split(",").map(s => s.trim())
      };

      // 1. Fetch SHAP Replay
      const replayRes = await fetch(`${API}/api/explain/${sessionId}/replay`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      });
      if (!replayRes.ok) throw new Error("Replay failed");
      const replayJson = await replayRes.json();
      setShapData(replayJson);

      // 2. Fetch Counterfactuals
      const cfRes = await fetch(`${API}/api/explain/${sessionId}/counterfactual`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      });
      if (!cfRes.ok) throw new Error("Counterfactuals failed");
      const cfJson = await cfRes.json();
      setCfData(cfJson);

      // 3. Fetch default 'manager' narrative
      fetchNarrative(replayJson, cfJson, payload.sensitive_attributes, "manager");

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchNarrative = async (sData: any, cData: any, sAttrs: string[], mode: ExplainMode) => {
    setCurrentMode(mode);
    if (narratives[mode]) return;
    setLoadingNarrative(true);
    try {
      const res = await fetch(`${API}/api/explain/${sessionId}/narrative`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          shap_data: sData, 
          counterfactual_data: cData,
          sensitive_attributes: sAttrs,
          mode 
        })
      });
      if (res.ok) {
        const json = await res.json();
        setNarratives(prev => ({ ...prev, [mode]: json.narrative }));
      }
    } finally {
      setLoadingNarrative(false);
    }
  };

  const renderNarrative = (n: NarrativeResponse) => {
    if (n.error) {
      return (
        <div className="text-red-400 p-4 bg-red-900/20 rounded-lg">
          <p className="font-semibold mb-2">Failed to parse structured response.</p>
          <pre className="text-xs whitespace-pre-wrap">{n.raw_text}</pre>
        </div>
      );
    }

    return (
      <div className="space-y-6 text-sm">
        {/* Headline & Verdict */}
        <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-xl">
          <h3 className="text-lg font-bold text-white mb-1">{n.headline}</h3>
          <p className="text-blue-300 font-medium">Outcome: {n.verdict}</p>
        </div>

        {/* Why Summary */}
        <div>
          <h4 className="font-semibold text-gray-200 mb-2 uppercase text-xs tracking-wider">Summary</h4>
          <p className="text-gray-400 leading-relaxed">{n.why_summary}</p>
        </div>

        {/* Top Factors */}
        <div>
          <h4 className="font-semibold text-gray-200 mb-3 uppercase text-xs tracking-wider">Top Driving Factors</h4>
          <div className="grid gap-3">
            {n.top_factors?.map((f, i) => (
              <div key={i} className="flex gap-3 items-start p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <div className="mt-0.5">
                  {f.direction.includes("increases") || f.direction.includes("rejection") ? (
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  )}
                </div>
                <div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="font-semibold text-gray-200">{f.feature}</span>
                    <span className="text-xs text-gray-500 font-mono">value: {f.original_value}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider ${
                      f.relative_importance === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-gray-700 text-gray-300'
                    }`}>{f.relative_importance}</span>
                  </div>
                  <p className="text-gray-400 text-xs">{f.commentary}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Counterfactuals */}
        <div>
          <h4 className="font-semibold text-gray-200 mb-2 uppercase text-xs tracking-wider">How to Change the Outcome</h4>
          <p className="text-gray-400 mb-3">{n.counterfactual_insights?.summary}</p>
          <div className="space-y-2">
            {n.counterfactual_insights?.examples?.map((ex, i) => (
              <div key={i} className="p-3 bg-gray-800/30 rounded-lg text-xs text-gray-300 border border-gray-700/30">
                <p className="mb-2 font-medium text-blue-300">{ex.description}</p>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(ex.changed_features).map(([feat, change]) => (
                    <span key={feat} className="font-mono bg-black/40 px-2 py-1 rounded">
                      <span className="text-gray-500">{feat}:</span> {change}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Fairness & Risk */}
        <div className="p-4 bg-amber-900/10 border border-amber-500/20 rounded-xl">
          <h4 className="font-semibold text-amber-500 mb-2 uppercase text-xs tracking-wider flex items-center gap-2">
            <ShieldAlert className="w-4 h-4" /> Fairness & Risk Context
          </h4>
          <div className="space-y-2 text-gray-300 text-xs leading-relaxed">
            <p><span className="text-gray-500">Uses Protected Attributes:</span> <span className="font-semibold text-amber-400">{n.fairness_and_risk?.uses_protected_attributes}</span></p>
            <p>{n.fairness_and_risk?.protected_attribute_impact}</p>
            {n.fairness_and_risk?.notes && <p className="italic text-gray-400">{n.fairness_and_risk.notes}</p>}
          </div>
        </div>

        {/* Recommended Actions */}
        <div>
          <h4 className="font-semibold text-gray-200 mb-2 uppercase text-xs tracking-wider">Recommended Actions</h4>
          <ul className="list-disc pl-5 space-y-1 text-gray-400">
            {n.recommended_actions?.map((action, i) => (
              <li key={i}>{action}</li>
            ))}
          </ul>
        </div>

      </div>
    );
  };

  // Render components
  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-end justify-between">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/15 mb-3">
                <BookOpen className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-[11px] uppercase tracking-widest font-semibold text-blue-400">Phase 5</span>
              </div>
              <h1 className="text-3xl font-bold text-white mb-2">Explainability Engine</h1>
              <p className="text-sm text-gray-500">Analyze individual predictions via SHAP, DiCE counterfactuals, and Persona-based LLM narratives.</p>
            </div>
            <div className="flex gap-4">
               <input type="text" placeholder="Target Col" value={targetCol} onChange={e=>setTargetCol(e.target.value)} className="px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm" />
               <input type="text" placeholder="Sensitive Attrs (comma sep)" value={sensitiveAttrs} onChange={e=>setSensitiveAttrs(e.target.value)} className="px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm w-48" />
               <div className="flex items-center gap-2">
                 <label className="text-sm text-gray-500">Row Index:</label>
                 <input type="number" min={0} value={rowIndex} onChange={e=>setRowIndex(parseInt(e.target.value))} className="w-20 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm" />
               </div>
               <button onClick={analyzeRow} disabled={loading} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow-lg flex items-center gap-2 disabled:opacity-50 transition-all">
                 {loading ? <Loader2 className="w-4 h-4 animate-spin"/> : <Play className="w-4 h-4"/>} Analyze Row
               </button>
            </div>
        </div>

        {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl flex items-center gap-2">
              <XCircle className="w-5 h-5" /> {error}
            </div>
        )}

        {/* Content area */}
        {shapData && (
          <div className="flex flex-col lg:flex-row gap-6">
            
            {/* Left Column: Math & Charts (SHAP + DiCE) */}
            <div className="lg:w-1/3 space-y-6">
              
              {/* 1. SHAP Waterfall equivalent */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
                <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><BrainCircuit className="w-5 h-5 text-blue-400"/> Decision Replay</h2>
                <div className="flex flex-col gap-2 text-sm text-gray-400 mb-6 bg-black/40 p-3 rounded-lg">
                   <div className="flex justify-between"><span>Prediction:</span> <strong className="text-white">{shapData.prediction}</strong></div>
                   <div className="flex justify-between"><span>Probability:</span> <strong className="text-blue-400">{(shapData.probability*100).toFixed(1)}%</strong></div>
                   <div className="flex justify-between"><span>Base Value:</span> <strong className="text-gray-300">{shapData.base_value.toFixed(4)}</strong></div>
                </div>
                
                <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-3 font-semibold">SHAP Contributions</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData.contributions.slice(0, 10)} layout="vertical" margin={{ left: 40, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                      <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 10 }} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: "#9ca3af", fontSize: 10 }} width={70} />
                      <Tooltip cursor={{fill: '#1f2937'}} contentStyle={{ backgroundColor: '#111827', border: 'none', borderRadius: '8px', fontSize: '12px' }} />
                      <ReferenceLine x={0} stroke="#4b5563" />
                      <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                        {shapData.contributions.slice(0, 10).map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#ef4444' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 2. Counterfactuals raw */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
                <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><Search className="w-5 h-5 text-emerald-400"/> DiCE Engine</h2>
                {cfData && cfData.counterfactuals?.length > 0 ? (
                  <div className="space-y-4">
                    <p className="text-xs text-gray-400">Raw generated scenarios to flip prediction to <strong className="text-emerald-400">{cfData.desired_class}</strong>:</p>
                    {cfData.counterfactuals.map((cf: any, i: number) => (
                      <div key={i} className="p-3 bg-black/40 border border-gray-800 rounded-xl text-xs">
                        <div className="font-semibold text-gray-300 mb-2">Scenario {i+1}</div>
                        <ul className="space-y-1">
                          {Object.entries(cf.changes).map(([feat, vals]: any) => (
                            <li key={feat} className="flex justify-between items-center text-gray-400">
                              <span className="truncate max-w-[100px]" title={feat}>{feat}</span>
                              <span><span className="text-red-400">{vals.from.toFixed(2)}</span> <span className="mx-1">→</span> <span className="text-emerald-400">{vals.to.toFixed(2)}</span></span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No counterfactuals found for this instance.</p>
                )}
              </div>

            </div>

            {/* Right Column: Structured LLM Narratives */}
            <div className="lg:w-2/3">
              <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 h-full flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2"><BookOpen className="w-5 h-5 text-amber-400"/> Structured Narrative</h2>
                </div>
                
                {/* Tabs */}
                <div className="flex gap-2 mb-6 bg-black/40 p-1.5 rounded-xl border border-gray-800">
                  {(["technical", "manager", "legal"] as ExplainMode[]).map(mode => (
                    <button key={mode} onClick={() => fetchNarrative(shapData, cfData, sensitiveAttrs.split(",").map(s=>s.trim()), mode)}
                      className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-semibold rounded-lg transition-all ${
                        currentMode === mode 
                          ? 'bg-blue-600 text-white shadow-lg' 
                          : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
                      }`}>
                      {mode === "technical" && <BrainCircuit className="w-4 h-4"/>}
                      {mode === "manager" && <Briefcase className="w-4 h-4"/>}
                      {mode === "legal" && <Gavel className="w-4 h-4"/>}
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </button>
                  ))}
                </div>

                <div className="flex-1 bg-black/20 rounded-xl p-2 relative overflow-hidden">
                  {loadingNarrative ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/50 backdrop-blur-sm z-10 text-gray-400">
                      <Loader2 className="w-8 h-8 animate-spin mb-4 text-blue-500"/>
                      <p>Generating {currentMode} narrative...</p>
                    </div>
                  ) : null}

                  <div className="h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                    {narratives[currentMode] ? (
                      renderNarrative(narratives[currentMode])
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-600 italic">
                        Select a mode to view narrative
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}

      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.1);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(75, 85, 99, 0.4);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(75, 85, 99, 0.6);
        }
      `}} />
    </div>
  );
}
