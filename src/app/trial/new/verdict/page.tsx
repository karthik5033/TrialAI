"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Gavel, AlertTriangle, CheckCircle2, XCircle, Download, RotateCcw, Scale, Loader2, Code2, FileDown, ArrowRight, Fingerprint } from "lucide-react";
import BiasFingerprint from "@/app/components/BiasFingerprint";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

const FADE_UP = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 100, damping: 15 } },
};

const METRIC_LABELS: Record<string, string> = {
  demographic_parity: "Demographic Parity",
  equal_opportunity: "Equal Opportunity",
  disparate_impact: "Disparate Impact",
};

export default function NewTrialVerdictPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<any>(null);
  const [retrainError, setRetrainError] = useState<string | null>(null);

  const [typedCode, setTypedCode] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [statusIdx, setStatusIdx] = useState(0);

  const statuses = [
    "Analyzing bias patterns...",
    "Injecting fairness constraints...",
    "Retraining model...",
    "Complete"
  ];

  useEffect(() => {
    if (retraining && statusIdx < 2) {
      const timer = setInterval(() => {
        setStatusIdx(prev => prev + 1);
      }, 800);
      return () => clearInterval(timer);
    }
  }, [retraining, statusIdx]);

  const startTyping = (fullText: string) => {
    setIsTyping(true);
    setTypedCode("");
    setStatusIdx(3);
    let i = 0;
    const interval = setInterval(() => {
      i += 150;
      setTypedCode(fullText.slice(0, i));
      if (i >= fullText.length) {
        clearInterval(interval);
        setTypedCode(fullText);
        setIsTyping(false);
      }
    }, 5);
  };


  // Analysis data
  const [analysis, setAnalysis] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState("Dataset");
  const [shapData, setShapData] = useState<any[]>([]);
  const [fairnessMetrics, setFairnessMetrics] = useState<any>({});
  const [verdict, setVerdict] = useState("GUILTY");
  const [modelType, setModelType] = useState("Unknown");
  const [codeAnalysis, setCodeAnalysis] = useState<any>(null);
  const [hasScript, setHasScript] = useState(false);
  const [fingerprintOpen, setFingerprintOpen] = useState(false);
  const [originalScript, setOriginalScript] = useState<string | null>(null);

  useEffect(() => {
    const rawAnalysis = localStorage.getItem("trialAnalysis");
    const rawName = localStorage.getItem("trialDatasetName");
    const rawSessionId = localStorage.getItem("trialSessionId");
    // Dedicated keys for retrain — always stored at upload time
    const rawTargetCol = localStorage.getItem("trialTargetColumn");
    const rawSensitiveAttrs = localStorage.getItem("trialSensitiveAttrs");

    if (!rawAnalysis) {
      router.push("/trial/upload");
      return;
    }

    try {
      const data = JSON.parse(rawAnalysis);
      // Merge dedicated backup keys so retrain always has these fields
      if (rawTargetCol && !data.target_column) data.target_column = rawTargetCol;
      if (rawSensitiveAttrs && !data.sensitive_attributes) {
        try { data.sensitive_attributes = JSON.parse(rawSensitiveAttrs); } catch {}
      }
      setAnalysis(data);
      setDatasetName(rawName || "Dataset");
      setSessionId(rawSessionId || data.session_id || null);
      setModelType(data.model_type || "Unknown");
      setVerdict(data.verdict || "GUILTY");
      setHasScript(!!data.code_analysis && data.code_analysis.risk_level !== "UNKNOWN");

      // Store the user's original script so LHS can show it immediately
      if (data.script_content) {
        setOriginalScript(data.script_content);
      }

      if (data.code_analysis) setCodeAnalysis(data.code_analysis);

      if (data.fairness_metrics) {
        setFairnessMetrics({
          demographic_parity: data.fairness_metrics.demographicParity ?? data.fairness_metrics.demographic_parity ?? 1,
          equal_opportunity: data.fairness_metrics.equalOpportunity ?? data.fairness_metrics.equal_opportunity ?? 1,
          disparate_impact: data.fairness_metrics.disparateImpact ?? data.fairness_metrics.disparate_impact ?? 1,
        });
      }

      if (data.shap_values) {
        setShapData(
          data.shap_values
            .map((sv: any) => ({
              feature: sv.is_proxy ? `${sv.feature} ⚠` : sv.feature,
              importance: sv.importance,
              isProxy: sv.is_proxy,
            }))
            .reverse()
        );
      }
    } catch (e) {
      console.error("Failed to parse analysis", e);
      router.push("/trial/upload");
      return;
    }

    setLoading(false);
  }, [router]);

  // ─── Retrain Handler ──────────────────────────────────────────────────
  const handleRetrain = async () => {
    if (!sessionId) {
      setRetrainError("No session ID found. Please re-run the analysis.");
      return;
    }
    setRetraining(true);
    setRetrainError(null);
    setStatusIdx(0);

    try {
      // Resolve target column & sensitive attrs — stored directly in the analysis object
      const targetCol = analysis?.target_column || analysis?.primary_protected_attribute || "";
      const sensitiveAttrs = analysis?.sensitive_attributes ?? 
                             (analysis?.primary_protected_attribute ? [analysis.primary_protected_attribute] : []);

      if (!targetCol) {
        throw new Error("Target column not found in analysis data. Please re-upload your dataset.");
      }

      const requestBody = { 
        strategy: "optimize",
        target_column: targetCol,
        sensitive_attributes: sensitiveAttrs
      };
      console.log("[Retrain] Sending:", requestBody, "sessionId:", sessionId);

      const res = await fetch(`/api/remediation/run/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      const rawText = await res.text();
      let initData;
      try {
        initData = JSON.parse(rawText);
      } catch {
        throw new Error(`Server returned invalid response: ${rawText.slice(0, 200)}`);
      }

      if (!res.ok) {
        throw new Error(initData.error || initData.detail || initData.rawResponse || "Failed to start retrain");
      }

      const runId = initData.run_id;
      if (!runId) {
        throw new Error("Backend did not return a run_id for polling.");
      }

      // --- Poll for status ---
      let isComplete = false;
      let finalData = null;
      let pollCount = 0;
      
      while (!isComplete && pollCount < 120) { // Max 4 minutes
        pollCount++;
        await new Promise(resolve => setTimeout(resolve, 2000)); // poll every 2s
        
        const statusRes = await fetch(`/api/remediation/status/${runId}`);
        const statusData = await statusRes.json();
        
        if (!statusRes.ok) {
          throw new Error(statusData.detail || "Failed to poll status");
        }
        
        if (statusData.status === "failed") {
          throw new Error(statusData.error || "Background task failed");
        }
        
        if (statusData.status === "complete") {
          isComplete = true;
          finalData = statusData.result;
          break;
        }
        
        // Optionally update UI with statusData.step if we had a state for it
      }

      if (!isComplete || !finalData) {
        throw new Error("Remediation timed out.");
      }

      if (finalData.retrain_success === false) {
        const errMsg = finalData.error || finalData.detail || "Retrain script failed";
        const stderr = finalData.stderr ? `\n\nScript error:\n${finalData.stderr.slice(-500)}` : "";
        setRetrainError(errMsg + stderr);
        return;
      }

      setRetrainResult(finalData);
      
      // Update the original script from the backend response if available
      if (finalData.original_script) {
        setOriginalScript(finalData.original_script);
      }
      if (finalData.modified_script) {
        startTyping(finalData.modified_script);
      }
    } catch (err: any) {
      setRetrainError(err.message || "Retrain failed.");
    } finally {
      setRetraining(false);
    }
  };

  // ─── PDF Download ─────────────────────────────────────────────────────
  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(20);
    doc.text("TrialAI - Bias Audit Report", 20, 25);
    doc.setFontSize(11);
    doc.text(`Dataset: ${datasetName}`, 20, 35);
    doc.text(`Model: ${modelType}`, 20, 42);
    doc.text(`Verdict: ${verdict}`, 20, 49);
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 20, 56);

    doc.setFontSize(14);
    doc.text("Fairness Metrics (Fairlearn)", 20, 70);

    autoTable(doc, {
      startY: 75,
      head: [["Metric", "Score", "Threshold", "Status"]],
      body: Object.entries(fairnessMetrics).map(([key, val]) => [
        METRIC_LABELS[key] || key,
        (val as number).toFixed(4),
        "> 0.80",
        (val as number) >= 0.8 ? "PASS" : "FAIL",
      ]),
    });

    let yPos = (doc as any).lastAutoTable?.finalY || (doc as any).autoTable?.previous?.finalY || 100;

    if (shapData.length > 0) {
      doc.setFontSize(14);
      doc.text("SHAP Feature Importance", 20, yPos);
      autoTable(doc, {
        startY: yPos + 5,
        head: [["Feature", "Importance"]],
        body: [...shapData].reverse().map(s => [s.feature, s.importance.toFixed(4)]),
      });
      yPos = (doc as any).lastAutoTable?.finalY || (doc as any).autoTable?.previous?.finalY || (yPos + 50);
    }

    if (retrainResult) {
      doc.setFontSize(14);
      doc.text("Retrain Results", 20, yPos);
      autoTable(doc, {
        startY: yPos + 5,
        head: [["Metric", "Before", "After", "Improvement"]],
        body: Object.keys(retrainResult.before || {}).map(key => [
          METRIC_LABELS[key] || key,
          retrainResult.before[key]?.toFixed(4),
          retrainResult.after[key]?.toFixed(4),
          `${retrainResult.improvement[key] > 0 ? "+" : ""}${retrainResult.improvement[key]?.toFixed(1)}%`,
        ]),
      });
    }

    doc.save(`TrialAI-Report-${datasetName}-${new Date().toISOString().slice(0, 10)}.pdf`);
  };

  if (loading) {
    return (
      <div className="relative min-h-screen bg-black flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-gold" />
      </div>
    );
  }

  const charges = Object.entries(fairnessMetrics).map(([key, val]) => ({
    name: METRIC_LABELS[key] || key,
    score: val as number,
    pass: (val as number) >= 0.8,
  }));

  return (
    <div className="relative min-h-screen text-white pt-16 font-sans selection:bg-white/20">
      {/* Video Background */}
      <div className="fixed inset-0 w-full h-full z-[-2]">
        <video autoPlay loop muted playsInline className="w-full h-full object-cover scale-105">
          <source src="/scales-video.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="fixed inset-0 bg-black/60 z-[-1]" />
      {/* PDF Report Bar */}
      <div className="border-b border-white/10 bg-white/[0.02] backdrop-blur-md">
        <div className="max-w-5xl mx-auto px-6 h-12 flex items-center justify-between">
          <span className="text-sm font-medium text-white/60">Trial Verdict — {datasetName}</span>
          <button onClick={downloadPDF}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium border border-white/10 rounded-lg hover:bg-black/40 backdrop-blur-2xl transition-colors">
            <Download className="w-4 h-4" /> PDF Report
          </button>
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-6 py-12">
        <motion.div initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.1 } } }} className="space-y-10">

          {/* Verdict Header */}
          <motion.div variants={FADE_UP} className="text-center">
            <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-2xl mb-6 ${verdict === "GUILTY" ? 'bg-white/[0.03] border border-white/10 shadow-2xl' : 'bg-white/[0.03] border border-white/10 shadow-2xl'}`}>
              <Gavel className={`w-8 h-8 ${verdict === "GUILTY" ? 'text-red-400' : 'text-green-400'}`} />
              <span className="text-4xl font-light tracking-tight text-white">
                {verdict}
              </span>
            </div>
            <h1 className="text-2xl font-bold mb-2">
              {verdict === "GUILTY" ? "Systematic Bias Confirmed" : "No Significant Bias Detected"}
            </h1>
            <p className="text-white/60 max-w-lg mx-auto">
              {modelType} model trained on <strong>{datasetName}</strong> — Audited with Fairlearn + SHAP
            </p>
          </motion.div>

          {/* Charges Grid */}
          <motion.div variants={FADE_UP} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {charges.map((charge) => (
              <div key={charge.name} className={`p-5 rounded-xl border ${charge.pass ? 'bg-white/[0.03] border-white/10 shadow-xl' : 'bg-white/[0.03] border-white/10 shadow-xl'}`}>
                <div className="flex items-center gap-2 mb-3">
                  {charge.pass
                    ? <CheckCircle2 className="w-5 h-5 text-green-400" />
                    : <XCircle className="w-5 h-5 text-red-400" />}
                  <span className="text-sm font-semibold">{charge.name}</span>
                </div>
                <div className="text-4xl font-light tracking-tight text-white">
                  {charge.score.toFixed(2)}
                </div>
                <p className={`text-xs mt-1 ${charge.pass ? 'text-green-400' : 'text-red-400'}`}>
                  Threshold: &gt; 0.80 — {charge.pass ? "PASSED" : "FAILED"}
                </p>
              </div>
            ))}
          </motion.div>

          {/* SHAP Chart */}
          {shapData.length > 0 && (
            <motion.div variants={FADE_UP} className="bg-black/40 backdrop-blur-2xl border border-white/10 rounded-xl p-6">
              <h2 className="text-lg font-bold mb-1">SHAP Feature Importance</h2>
              <p className="text-sm text-white/60 mb-6">Real SHAP values computed on your model. Red bars indicate proxy features correlated with sensitive attributes.</p>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#ffffff1a" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#ffffff99" }} width={130} />
                    <Tooltip contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={20}>
                      {shapData.map((entry, i) => (
                        <Cell key={`cell-${i}`} fill={entry.isProxy ? "#EF4444" : "#3B82F6"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}

          {/* Code Analysis */}
          {codeAnalysis && codeAnalysis.risk_level !== "UNKNOWN" && (
            <motion.div variants={FADE_UP} className="bg-black/40 backdrop-blur-2xl border border-white/10 rounded-xl p-6">
              <h2 className="text-lg font-bold mb-1 flex items-center gap-2">
                <Code2 className="w-5 h-5" /> Training Code Analysis
              </h2>
              <p className="text-sm text-white/60 mb-4">LLM-powered analysis of your training script for bias-prone patterns.</p>

              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-bold mb-4 ${
                codeAnalysis.risk_level === 'HIGH' ? 'bg-red-500/20 text-white' :
                codeAnalysis.risk_level === 'MODERATE' ? 'bg-gold/20 text-white' : 'bg-green-500/20 text-white'
              }`}>
                {codeAnalysis.risk_level} RISK
              </div>

              {codeAnalysis.issues?.length > 0 && (
                <div className="space-y-2 mb-4">
                  {codeAnalysis.issues.map((issue: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <XCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                      <span>{issue}</span>
                    </div>
                  ))}
                </div>
              )}
              {codeAnalysis.recommendations?.length > 0 && (
                <div className="space-y-2">
                  {codeAnalysis.recommendations.map((rec: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* Retrain Section */}
          {verdict === "GUILTY" && (
            <motion.div variants={FADE_UP} className="bg-black/40 backdrop-blur-2xl border border-white/10 rounded-xl overflow-hidden shadow-lg">
              <div className="p-6 border-b border-white/10 bg-black/40 backdrop-blur-2xl">
                <h2 className="text-lg font-bold flex items-center gap-2 text-white">
                  <RotateCcw className="w-5 h-5" /> Court Reform Order — Automated Retraining
                </h2>
                <p className="text-sm text-white/60 mt-1">
                  {hasScript
                    ? "Your training script will be automatically modified by AI to inject fairness constraints, then re-executed to produce a bias-mitigated model."
                    : "No training script was uploaded. The system will auto-generate a training script based on your model type and apply fairness constraints."}
                </p>
              </div>

              {!retrainResult && !retraining && !isTyping && (
                <div className="p-6 bg-black/40 backdrop-blur-2xl">
                  <button
                    onClick={handleRetrain}
                    disabled={!sessionId}
                    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                      sessionId
                        ? 'bg-foreground text-background hover:bg-foreground/90 shadow-md'
                        : 'bg-white/5 text-white/50 cursor-not-allowed'
                    }`}
                  >
                    <RotateCcw className="w-5 h-5" />
                    {hasScript ? "Run Mitigated Retrain" : "Auto-Generate & Retrain"}
                  </button>
                </div>
              )}

              {retrainError && (
                <div className="p-6 bg-black/40 backdrop-blur-2xl border-t border-white/10">
                  <div className="p-4 bg-white/[0.03] border border-white/10 shadow-2xl rounded-lg">
                    <p className="text-sm font-semibold text-white">Retrain Failed</p>
                    <pre className="text-xs text-red-400 mt-2 max-h-[200px] overflow-y-auto whitespace-pre-wrap font-mono bg-red-500/20/50 rounded p-2">{retrainError}</pre>
                    <button
                      onClick={() => { setRetrainError(null); handleRetrain(); }}
                      className="mt-3 flex items-center gap-2 px-4 py-2 text-sm font-medium bg-foreground text-background rounded-lg hover:bg-foreground/90 transition-colors"
                    >
                      <RotateCcw className="w-4 h-4" /> Retry
                    </button>
                  </div>
                </div>
              )}

              {(retraining || retrainResult || isTyping) && !retrainError && (
                <div className="flex flex-col border-b border-[#30363d] bg-[#0d1117]">
                  {/* Split Panels */}
                  <div className="grid grid-cols-1 md:grid-cols-2 text-gray-300 font-mono text-xs md:divide-x divide-y md:divide-y-0 divide-[#30363d] h-[450px]">
                    {/* Left Panel */}
                    <div className="overflow-y-auto flex flex-col relative bg-[#0d1117]">
                      <div className="sticky top-0 bg-[#0d1117] border-b border-[#30363d] px-4 py-2 font-semibold text-gray-400 flex justify-between z-10 shadow-sm">
                         <span>original_script.py</span>
                      </div>
                      <div className="py-2 pb-8">
                        {(originalScript || "# No training script was uploaded.\n# The system will auto-generate one for retraining.")?.split('\n').map((line: string, i: number) => {
                          const newLines = retrainResult?.modified_script?.split('\n').map((l: string) => l.trim()) || [];
                          const isRemoved = retrainResult && !newLines.includes(line.trim()) && line.trim() !== "";
                          return (
                            <div key={i} className={`px-4 flex leading-relaxed ${isRemoved ? "bg-red-900/30 text-red-200" : ""}`}>
                              <span className="text-[#6e7681] select-none w-6 text-right mr-4 shrink-0">{i + 1}</span>
                              <span className="whitespace-pre-wrap break-all">{line || " "}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                    {/* Right Panel */}
                    <div className="overflow-y-auto flex flex-col relative bg-[#0d1117]">
                      <div className="sticky top-0 bg-[#0d1117] border-b border-[#30363d] px-4 py-2 font-semibold text-[#4493f8] flex justify-between z-10 shadow-sm">
                         <span>mitigated_script.py</span>
                      </div>
                      <div className="py-2 pb-8">
                        {!isTyping && !retrainResult ? (
                          <div className="px-4 text-[#8b949e] italic mt-2 animate-pulse">Waiting for AI to analyze and modify the script...</div>
                        ) : (
                          typedCode.split('\n').map((line: string, i: number, arr: string[]) => {
                            const oldLines = (originalScript || "")?.split('\n').map((l: string) => l.trim()) || [];
                            const isAdded = !oldLines.includes(line.trim()) && line.trim() !== "";
                            return (
                              <div key={i} className={`px-4 flex leading-relaxed ${isAdded ? "bg-green-900/30 text-green-200" : ""}`}>
                                <span className="text-[#6e7681] select-none w-6 text-right mr-4 shrink-0">{i + 1}</span>
                                <span className="whitespace-pre-wrap break-all">{line || " "}</span>
                                {i === arr.length - 1 && isTyping && (
                                   <span className="inline-block w-2 h-3 bg-gray-400 animate-pulse ml-1 align-middle" />
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {/* Status Bar */}
                  <div className="bg-[#161b22] px-4 py-3 flex items-center gap-3 text-xs font-mono border-t border-[#30363d]">
                     {retraining ? (
                       <>
                         <Loader2 className="w-4 h-4 animate-spin text-[#4493f8]" />
                         <span className="text-[#4493f8]">{statuses[statusIdx]}</span>
                       </>
                     ) : isTyping ? (
                       <>
                         <Loader2 className="w-4 h-4 animate-spin text-[#3fb950]" />
                         <span className="text-[#3fb950]">Applying changes character by character...</span>
                       </>
                     ) : (
                       <>
                         <CheckCircle2 className="w-4 h-4 text-[#3fb950]" />
                         <span className="text-[#3fb950]">Complete</span>
                       </>
                     )}
                  </div>
                </div>
              )}

              {retrainResult && !isTyping && !retrainError && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="p-6 bg-black/40 backdrop-blur-2xl space-y-6">
                  {/* Metrics Table */}
                  <div className="bg-transparent border border-white/10 rounded-xl overflow-hidden">
                    <div className="bg-black/40 backdrop-blur-2xl/5 px-4 py-3 border-b border-white/10">
                      <h3 className="font-semibold text-sm flex items-center gap-2">
                        {retrainResult.retrial_passed
                          ? <><CheckCircle2 className="w-4 h-4 text-green-400" /> Retrial PASSED</>
                          : <><XCircle className="w-4 h-4 text-gold" /> Retrial Partial</>}
                      </h3>
                    </div>
                    <table className="w-full text-sm">
                      <thead className="bg-black/40 backdrop-blur-2xl/[0.02] text-xs uppercase text-white/60">
                        <tr>
                          <th className="px-4 py-2 text-left">Metric</th>
                          <th className="px-4 py-2 text-right">Before</th>
                          <th className="px-4 py-2 text-center"></th>
                          <th className="px-4 py-2 text-right">After</th>
                          <th className="px-4 py-2 text-right">Change</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {Object.keys(retrainResult.before || {}).map(key => (
                          <tr key={key}>
                            <td className="px-4 py-3 font-medium">{METRIC_LABELS[key] || key}</td>
                            <td className="px-4 py-3 text-right font-mono text-red-400">{retrainResult.before[key]?.toFixed(4)}</td>
                            <td className="px-4 py-3 text-center"><ArrowRight className="w-4 h-4 text-white/30 mx-auto" /></td>
                            <td className={`px-4 py-3 text-right font-mono ${retrainResult.after[key] >= 0.8 ? 'text-green-400' : 'text-gold'}`}>
                              {retrainResult.after[key]?.toFixed(4)}
                            </td>
                            <td className={`px-4 py-3 text-right font-mono font-bold ${retrainResult.improvement[key] > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {retrainResult.improvement[key] > 0 ? "+" : ""}{retrainResult.improvement[key]?.toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Accuracy comparison */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-transparent border border-white/10 rounded-xl text-center">
                      <p className="text-xs text-white/50 uppercase font-bold mb-1">Original Accuracy</p>
                      <p className="text-2xl font-bold">{((retrainResult.original_accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-4 bg-transparent border border-white/10 rounded-xl text-center">
                      <p className="text-xs text-white/50 uppercase font-bold mb-1">New Accuracy</p>
                      <p className="text-2xl font-bold text-blue-400">{((retrainResult.mitigated_accuracy || retrainResult.new_accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  {/* Downloads */}
                  <div className="flex flex-wrap gap-3">
                    {sessionId && (
                      <>
                        <a
                          href={`/api/download/${sessionId}/model`}
                          className="flex items-center gap-2 px-5 py-2.5 bg-foreground text-background rounded-lg font-medium text-sm hover:bg-foreground/90 transition-colors"
                        >
                          <FileDown className="w-4 h-4" /> Download Mitigated Model
                        </a>
                        <a
                          href={`/api/download/${sessionId}/script`}
                          className="flex items-center gap-2 px-5 py-2.5 border border-white/10 rounded-lg font-medium text-sm hover:bg-black/40 backdrop-blur-2xl transition-colors"
                        >
                          <Code2 className="w-4 h-4" /> Download Modified Script
                        </a>
                      </>
                    )}
                    <button onClick={downloadPDF}
                      className="flex items-center gap-2 px-5 py-2.5 border border-white/10 rounded-lg font-medium text-sm hover:bg-black/40 backdrop-blur-2xl transition-colors">
                      <Download className="w-4 h-4" /> Download PDF Report
                    </button>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Footer Actions */}
          <motion.div variants={FADE_UP} className="flex justify-center gap-4 pt-4 pb-12">
            <button
              onClick={() => setFingerprintOpen(true)}
              className="flex items-center gap-2 px-6 py-3 bg-foreground text-background rounded-lg font-medium text-sm hover:bg-foreground/90 transition-all shadow-sm"
            >
              <Fingerprint className="w-4 h-4" /> View Bias Fingerprint
            </button>
            <Link href="/trial/upload"
              className="flex items-center gap-2 px-6 py-3 border border-white/10 rounded-lg font-medium text-sm hover:bg-black/40 backdrop-blur-2xl transition-colors">
              <RotateCcw className="w-4 h-4" /> New Trial
            </Link>
            <Link href="/"
              className="flex items-center gap-2 px-6 py-3 bg-foreground text-background rounded-lg font-medium text-sm hover:bg-foreground/90 transition-colors">
              Return to Home
            </Link>
          </motion.div>

          {/* Bias Fingerprint Modal */}
          <BiasFingerprint
            open={fingerprintOpen}
            onClose={() => setFingerprintOpen(false)}
            datasetName={datasetName}
            sensitiveAttr={analysis?.sensitive_attributes?.join(", ") || "unknown"}
            demographicParity={fairnessMetrics.demographic_parity || 0}
            equalOpportunity={fairnessMetrics.equal_opportunity || 0}
            disparateImpact={fairnessMetrics.disparate_impact || 0}
          />

        </motion.div>
      </main>
    </div>
  );
}
