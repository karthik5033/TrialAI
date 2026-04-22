"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from "recharts";
import { 
  Search, BrainCircuit, ShieldAlert, BookOpen, Loader2, Play, Users, Briefcase, Gavel, XCircle
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ExplainMode = "technical" | "manager" | "legal";

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
  const [narratives, setNarratives] = useState<Record<string, string>>({});
  const [loadingNarrative, setLoadingNarrative] = useState(false);

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
      setCfData(await cfRes.json());

      // 3. Fetch default 'manager' narrative
      fetchNarrative(replayJson, "manager");

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchNarrative = async (data: any, mode: ExplainMode) => {
    if (narratives[mode]) return;
    setLoadingNarrative(true);
    try {
      const res = await fetch(`${API}/api/explain/${sessionId}/narrative`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shap_data: data, mode })
      });
      if (res.ok) {
        const json = await res.json();
        setNarratives(prev => ({ ...prev, [mode]: json.narrative }));
      }
    } finally {
      setLoadingNarrative(false);
    }
  };

  // Render components
  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-end justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Decision Explainability</h1>
              <p className="text-gray-500">Analyze individual predictions via SHAP, DiCE counterfactuals, and LLM narratives.</p>
            </div>
            <div className="flex gap-4">
               <input type="text" placeholder="Target Col" value={targetCol} onChange={e=>setTargetCol(e.target.value)} className="px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm" />
               <input type="text" placeholder="Sensitive Attrs (comma sep)" value={sensitiveAttrs} onChange={e=>setSensitiveAttrs(e.target.value)} className="px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm w-48" />
               <div className="flex items-center gap-2">
                 <label className="text-sm text-gray-500">Row Index:</label>
                 <input type="number" min={0} value={rowIndex} onChange={e=>setRowIndex(parseInt(e.target.value))} className="w-20 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm" />
               </div>
               <button onClick={analyzeRow} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow-lg flex items-center gap-2">
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* 1. SHAP Waterfall equivalent */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><BrainCircuit className="w-5 h-5 text-blue-400"/> Decision Replay (SHAP)</h2>
              <div className="flex justify-between text-sm text-gray-400 mb-6 bg-black/40 p-3 rounded-lg">
                 <span>Prediction: <strong className="text-white">{shapData.prediction}</strong></span>
                 <span>Probability: <strong className="text-blue-400">{(shapData.probability*100).toFixed(1)}%</strong></span>
                 <span>Base Value: <strong className="text-gray-300">{shapData.base_value.toFixed(4)}</strong></span>
              </div>
              
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData.contributions.slice(0, 10)} layout="vertical" margin={{ left: 50, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 12 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fill: "#9ca3af", fontSize: 11 }} width={80} />
                    <Tooltip cursor={{fill: '#1f2937'}} contentStyle={{ backgroundColor: '#111827', border: 'none', borderRadius: '8px' }} />
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

            <div className="space-y-6">
              {/* 2. Counterfactuals */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
                <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><Search className="w-5 h-5 text-emerald-400"/> DiCE Counterfactuals</h2>
                {cfData && cfData.counterfactuals?.length > 0 ? (
                  <div className="space-y-4">
                    <p className="text-sm text-gray-400">Minimum changes required to flip prediction from <strong className="text-red-400">{cfData.original_prediction}</strong> to <strong className="text-emerald-400">{cfData.desired_class}</strong>:</p>
                    {cfData.counterfactuals.map((cf: any, i: number) => (
                      <div key={i} className="p-4 bg-black/40 border border-gray-800 rounded-xl text-sm">
                        <div className="font-semibold text-gray-300 mb-2">Scenario {i+1}</div>
                        <ul className="space-y-1">
                          {Object.entries(cf.changes).map(([feat, vals]: any) => (
                            <li key={feat} className="flex justify-between items-center text-gray-400">
                              <span>{feat}</span>
                              <span><span className="text-red-400">{vals.from.toFixed(2)}</span> <span className="mx-2">→</span> <span className="text-emerald-400">{vals.to.toFixed(2)}</span></span>
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

              {/* 3. LLM Narratives */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 flex-1 flex flex-col">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-bold text-white flex items-center gap-2"><BookOpen className="w-5 h-5 text-amber-400"/> Explainability Narrative</h2>
                </div>
                
                {/* Tabs */}
                <div className="flex gap-2 mb-4 bg-black/40 p-1 rounded-lg">
                  {(["manager", "technical", "legal"] as ExplainMode[]).map(mode => (
                    <button key={mode} onClick={() => fetchNarrative(shapData, mode)}
                      className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-md transition-colors ${narratives[mode] ? 'bg-gray-800 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                      {mode === "manager" && <Briefcase className="w-3.5 h-3.5"/>}
                      {mode === "technical" && <BrainCircuit className="w-3.5 h-3.5"/>}
                      {mode === "legal" && <Gavel className="w-3.5 h-3.5"/>}
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </button>
                  ))}
                </div>

                <div className="flex-1 bg-black/40 border border-gray-800 rounded-xl p-5 text-sm text-gray-300 leading-relaxed overflow-y-auto max-h-64">
                  {loadingNarrative ? (
                    <div className="flex items-center justify-center h-full text-gray-500 gap-2"><Loader2 className="w-4 h-4 animate-spin"/> Claude is analyzing...</div>
                  ) : (
                    <div className="whitespace-pre-wrap">{narratives["manager"] || narratives["technical"] || narratives["legal"] || "Select a mode to generate narrative."}</div>
                  )}
                </div>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
