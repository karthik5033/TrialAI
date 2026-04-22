"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Gavel, AlertTriangle, CheckCircle2, XCircle, Download, RotateCcw, Scale, Loader2, Code2, FileDown, ArrowRight, Fingerprint } from "lucide-react";
import BiasFingerprint from "@/app/components/BiasFingerprint";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Link from "next/link";
import { useRouter } from "next/navigation";
import jsPDF from "jspdf";
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

  useEffect(() => {
    const rawAnalysis = localStorage.getItem("trialAnalysis");
    const rawName = localStorage.getItem("trialDatasetName");
    const rawSessionId = localStorage.getItem("trialSessionId");

    if (!rawAnalysis) {
      router.push("/upload");
      return;
    }

    try {
      const data = JSON.parse(rawAnalysis);
      setAnalysis(data);
      setDatasetName(rawName || "Dataset");
      setSessionId(rawSessionId || data.session_id || null);
      setModelType(data.model_type || "Unknown");
      setVerdict(data.verdict || "GUILTY");
      setHasScript(!!data.code_analysis && data.code_analysis.risk_level !== "UNKNOWN");

      if (data.code_analysis) setCodeAnalysis(data.code_analysis);

      if (data.fairness_metrics) {
        setFairnessMetrics({
          demographic_parity: data.fairness_metrics.demographic_parity ?? 1,
          equal_opportunity: data.fairness_metrics.equal_opportunity ?? 1,
          disparate_impact: data.fairness_metrics.disparate_impact ?? 1,
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
      router.push("/upload");
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

    try {
      const res = await fetch("/api/mitigate-and-retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      const rawText = await res.text();
      let data;
      try {
        data = JSON.parse(rawText);
      } catch {
        throw new Error(`Server returned invalid response: ${rawText.slice(0, 200)}`);
      }

      if (!res.ok && !data.retrain_success && data.retrain_success !== false) {
        throw new Error(data.error || data.rawResponse || "Retrain failed");
      }

      if (data.retrain_success === false) {
        // Script execution failed -- show error with details
        const errMsg = data.error || "Retrain script failed";
        const stderr = data.stderr ? `\n\nScript error:\n${data.stderr.slice(-500)}` : "";
        setRetrainError(errMsg + stderr);
        return;
      }

      setRetrainResult(data);
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

    let yPos = (doc as any).lastAutoTable.finalY + 15;

    if (shapData.length > 0) {
      doc.setFontSize(14);
      doc.text("SHAP Feature Importance", 20, yPos);
      autoTable(doc, {
        startY: yPos + 5,
        head: [["Feature", "Importance"]],
        body: [...shapData].reverse().map(s => [s.feature, s.importance.toFixed(4)]),
      });
      yPos = (doc as any).lastAutoTable.finalY + 15;
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
      <div className="min-h-screen bg-background flex items-center justify-center">
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
    <div className="min-h-screen bg-background text-foreground">
      {/* PDF Report Bar */}
      <div className="border-b border-border bg-surface">
        <div className="max-w-5xl mx-auto px-6 h-12 flex items-center justify-between">
          <span className="text-sm font-medium text-foreground/60">Trial Verdict — {datasetName}</span>
          <button onClick={downloadPDF}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium border border-border rounded-lg hover:bg-surface transition-colors">
            <Download className="w-4 h-4" /> PDF Report
          </button>
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-6 py-12">
        <motion.div initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.1 } } }} className="space-y-10">

          {/* Verdict Header */}
          <motion.div variants={FADE_UP} className="text-center">
            <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-2xl mb-6 ${verdict === "GUILTY" ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
              <Gavel className={`w-8 h-8 ${verdict === "GUILTY" ? 'text-red-600' : 'text-green-600'}`} />
              <span className={`text-3xl font-black tracking-tight ${verdict === "GUILTY" ? 'text-red-700' : 'text-green-700'}`}>
                {verdict}
              </span>
            </div>
            <h1 className="text-2xl font-bold mb-2">
              {verdict === "GUILTY" ? "Systematic Bias Confirmed" : "No Significant Bias Detected"}
            </h1>
            <p className="text-foreground/60 max-w-lg mx-auto">
              {modelType} model trained on <strong>{datasetName}</strong> — Audited with Fairlearn + SHAP
            </p>
          </motion.div>

          {/* Charges Grid */}
          <motion.div variants={FADE_UP} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {charges.map((charge) => (
              <div key={charge.name} className={`p-5 rounded-xl border ${charge.pass ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                <div className="flex items-center gap-2 mb-3">
                  {charge.pass
                    ? <CheckCircle2 className="w-5 h-5 text-green-600" />
                    : <XCircle className="w-5 h-5 text-red-600" />}
                  <span className="text-sm font-semibold">{charge.name}</span>
                </div>
                <div className={`text-3xl font-black ${charge.pass ? 'text-green-700' : 'text-red-700'}`}>
                  {charge.score.toFixed(2)}
                </div>
                <p className={`text-xs mt-1 ${charge.pass ? 'text-green-600' : 'text-red-600'}`}>
                  Threshold: &gt; 0.80 — {charge.pass ? "PASSED" : "FAILED"}
                </p>
              </div>
            ))}
          </motion.div>

          {/* SHAP Chart */}
          {shapData.length > 0 && (
            <motion.div variants={FADE_UP} className="bg-surface border border-border rounded-xl p-6">
              <h2 className="text-lg font-bold mb-1">SHAP Feature Importance</h2>
              <p className="text-sm text-foreground/60 mb-6">Real SHAP values computed on your model. Red bars indicate proxy features correlated with sensitive attributes.</p>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#64748B" }} width={130} />
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
            <motion.div variants={FADE_UP} className="bg-surface border border-border rounded-xl p-6">
              <h2 className="text-lg font-bold mb-1 flex items-center gap-2">
                <Code2 className="w-5 h-5" /> Training Code Analysis
              </h2>
              <p className="text-sm text-foreground/60 mb-4">LLM-powered analysis of your training script for bias-prone patterns.</p>

              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-bold mb-4 ${
                codeAnalysis.risk_level === 'HIGH' ? 'bg-red-100 text-red-700' :
                codeAnalysis.risk_level === 'MODERATE' ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'
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
            <motion.div variants={FADE_UP} className="bg-surface border border-border rounded-xl p-6">
              <h2 className="text-lg font-bold mb-1 flex items-center gap-2">
                <RotateCcw className="w-5 h-5" /> Court Reform Order — Automated Retraining
              </h2>
              <p className="text-sm text-foreground/60 mb-6">
                {hasScript
                  ? "Your training script will be automatically modified by AI to inject fairness constraints, then re-executed to produce a bias-mitigated model."
                  : "No training script was uploaded. The system will auto-generate a training script based on your model type and apply fairness constraints."}
              </p>

              {!retrainResult && !retraining && (
                <button
                  onClick={handleRetrain}
                  disabled={!sessionId}
                  className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                    sessionId
                      ? 'bg-foreground text-background hover:bg-foreground/90 shadow-md'
                      : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  <RotateCcw className="w-5 h-5" />
                  {hasScript ? "Run Mitigated Retrain" : "Auto-Generate & Retrain"}
                </button>
              )}

              {retraining && (
                <div className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                  <div>
                    <p className="text-sm font-semibold text-blue-800">Retraining in progress...</p>
                    <p className="text-xs text-blue-600">AI is modifying your training script and re-executing it with fairness constraints.</p>
                  </div>
                </div>
              )}

              {retrainError && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg mt-4">
                  <p className="text-sm font-semibold text-red-700">Retrain Failed</p>
                  <pre className="text-xs text-red-600 mt-2 max-h-[200px] overflow-y-auto whitespace-pre-wrap font-mono bg-red-100/50 rounded p-2">{retrainError}</pre>
                  <button
                    onClick={() => { setRetrainError(null); handleRetrain(); }}
                    className="mt-3 flex items-center gap-2 px-4 py-2 text-sm font-medium bg-foreground text-background rounded-lg hover:bg-foreground/90 transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" /> Retry
                  </button>
                </div>
              )}

              {retrainResult && (
                <div className="space-y-6 mt-4">
                  {/* Before / After Metrics */}
                  <div className="bg-background border border-border rounded-xl overflow-hidden">
                    <div className="bg-surface/50 px-4 py-3 border-b border-border">
                      <h3 className="font-semibold text-sm flex items-center gap-2">
                        {retrainResult.retrial_passed
                          ? <><CheckCircle2 className="w-4 h-4 text-green-600" /> Retrial PASSED</>
                          : <><XCircle className="w-4 h-4 text-amber-600" /> Retrial Partial</>}
                      </h3>
                    </div>
                    <table className="w-full text-sm">
                      <thead className="bg-surface/30 text-xs uppercase text-foreground/60">
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
                            <td className="px-4 py-3 text-right font-mono text-red-600">{retrainResult.before[key]?.toFixed(4)}</td>
                            <td className="px-4 py-3 text-center"><ArrowRight className="w-4 h-4 text-foreground/30 mx-auto" /></td>
                            <td className={`px-4 py-3 text-right font-mono ${retrainResult.after[key] >= 0.8 ? 'text-green-600' : 'text-amber-600'}`}>
                              {retrainResult.after[key]?.toFixed(4)}
                            </td>
                            <td className={`px-4 py-3 text-right font-mono font-bold ${retrainResult.improvement[key] > 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {retrainResult.improvement[key] > 0 ? "+" : ""}{retrainResult.improvement[key]?.toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Accuracy comparison */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-background border border-border rounded-xl text-center">
                      <p className="text-xs text-foreground/50 uppercase font-bold mb-1">Original Accuracy</p>
                      <p className="text-2xl font-bold">{((retrainResult.original_accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-4 bg-background border border-border rounded-xl text-center">
                      <p className="text-xs text-foreground/50 uppercase font-bold mb-1">New Accuracy</p>
                      <p className="text-2xl font-bold text-blue-600">{((retrainResult.new_accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  {/* Code Diff */}
                  {retrainResult.modified_script && (
                    <div className="bg-background border border-border rounded-xl overflow-hidden">
                      <div className="bg-surface/50 px-4 py-3 border-b border-border flex items-center justify-between">
                        <h3 className="font-semibold text-sm flex items-center gap-2">
                          <Code2 className="w-4 h-4" /> Modified Training Script
                        </h3>
                      </div>
                      <pre className="p-4 text-xs overflow-x-auto max-h-[300px] overflow-y-auto font-mono bg-gray-50 text-gray-800 leading-relaxed">
                        {retrainResult.modified_script}
                      </pre>
                    </div>
                  )}

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
                          className="flex items-center gap-2 px-5 py-2.5 border border-border rounded-lg font-medium text-sm hover:bg-surface transition-colors"
                        >
                          <Code2 className="w-4 h-4" /> Download Modified Script
                        </a>
                      </>
                    )}
                    <button onClick={downloadPDF}
                      className="flex items-center gap-2 px-5 py-2.5 border border-border rounded-lg font-medium text-sm hover:bg-surface transition-colors">
                      <Download className="w-4 h-4" /> Download PDF Report
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Footer Actions */}
          <motion.div variants={FADE_UP} className="flex justify-center gap-4 pt-4 pb-12">
            <button
              onClick={() => setFingerprintOpen(true)}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-lg font-medium text-sm hover:from-violet-700 hover:to-indigo-700 transition-all shadow-md"
            >
              <Fingerprint className="w-4 h-4" /> View Bias Fingerprint
            </button>
            <Link href="/upload"
              className="flex items-center gap-2 px-6 py-3 border border-border rounded-lg font-medium text-sm hover:bg-surface transition-colors">
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
