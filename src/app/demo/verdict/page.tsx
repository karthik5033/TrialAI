"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Gavel, AlertTriangle, CheckCircle2, XCircle, Download, RotateCcw, Scale, Loader2, PlayCircle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Link from "next/link";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

// --- FALLBACK DATA ---

const FALLBACK_SHAP = [
  { feature: "Prior Arrests", importance: 0.85 },
  { feature: "Age", importance: 0.65 },
  { feature: "Zipcode (Proxy)", importance: 0.55 },
  { feature: "Charge Degree", importance: 0.45 },
  { feature: "Employment Duration", importance: 0.35 },
  { feature: "Education Level", importance: 0.25 },
  { feature: "Marital Status", importance: 0.15 },
  { feature: "Substance Abuse History", importance: 0.12 },
].reverse();

const FALLBACK_FAIRNESS = {
  demographic_parity: 0.62,
  equal_opportunity: 0.75,
  disparate_impact: 0.58,
};

const FALLBACK_MITIGATION = [
  { metric: "Demographic Parity", before: "0.62", after: "0.85", diff: "+37%" },
  { metric: "Equal Opportunity", before: "0.75", after: "0.92", diff: "+22%" },
  { metric: "Disparate Impact", before: "0.58", after: "0.81", diff: "+39%" },
];

const JURY_PERSONAS = [
  { id: 1, name: "Marcus T.", age: 24, occupation: "Retail", demographic: "African American", outcome: "Denied" },
  { id: 2, name: "Sarah J.", age: 31, occupation: "Teacher", demographic: "Caucasian", outcome: "Approved" },
  { id: 3, name: "Luis M.", age: 28, occupation: "Construction", demographic: "Hispanic", outcome: "Denied" },
  { id: 4, name: "Emily R.", age: 45, occupation: "Manager", demographic: "Caucasian", outcome: "Approved" },
  { id: 5, name: "David K.", age: 22, occupation: "Student", demographic: "African American", outcome: "Denied" },
  { id: 6, name: "Anna C.", age: 38, occupation: "Nurse", demographic: "Asian", outcome: "Approved" },
  { id: 7, name: "James W.", age: 29, occupation: "Mechanic", demographic: "Caucasian", outcome: "Approved" },
  { id: 8, name: "Maria S.", age: 34, occupation: "Chef", demographic: "Hispanic", outcome: "Denied" },
  { id: 9, name: "Kevin B.", age: 41, occupation: "Accountant", demographic: "African American", outcome: "Approved" },
  { id: 10, name: "Rachel P.", age: 27, occupation: "Designer", demographic: "Caucasian", outcome: "Approved" },
  { id: 11, name: "Thomas L.", age: 50, occupation: "Driver", demographic: "Hispanic", outcome: "Denied" },
  { id: 12, name: "Jessica H.", age: 33, occupation: "Sales", demographic: "African American", outcome: "Denied" },
];

const FADE_UP_ANIMATION_VARIANTS = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 100, damping: 15 } },
};

// Metric key to display name mapping
const METRIC_LABELS: Record<string, string> = {
  demographic_parity: "Demographic Parity",
  equal_opportunity: "Equal Opportunity",
  disparate_impact: "Disparate Impact",
};

export default function DemoVerdictPage() {
  const trialId = "DEMO-1024";

  const [loading, setLoading] = useState(true);
  const [shapData, setShapData] = useState(FALLBACK_SHAP);
  const [fairnessMetrics, setFairnessMetrics] = useState(FALLBACK_FAIRNESS);
  const [mitigationData, setMitigationData] = useState(FALLBACK_MITIGATION);
  const datasetName = "COMPAS";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [analyzeRes, mitigateRes] = await Promise.all([
          fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_name: "COMPAS" }),
          }),
          fetch("/api/mitigate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_name: "COMPAS", technique: "reweighting" }),
          }),
        ]);

        if (analyzeRes.ok) {
          const analyzeData = await analyzeRes.json();

          if (analyzeData.fairness_metrics) {
            setFairnessMetrics({
              demographic_parity: Number(analyzeData.fairness_metrics.demographic_parity ?? analyzeData.fairness_metrics.demographicParity ?? 1.0),
              equal_opportunity: Number(analyzeData.fairness_metrics.equal_opportunity ?? analyzeData.fairness_metrics.equalOpportunity ?? 1.0),
              disparate_impact: Number(analyzeData.fairness_metrics.disparate_impact ?? analyzeData.fairness_metrics.disparateImpact ?? 1.0),
            });
          }

          if (analyzeData.shap_values) {
            setShapData(
              analyzeData.shap_values
                .map((sv: any) => ({
                  feature: sv.is_proxy ? `${sv.feature} (Proxy)` : sv.feature,
                  importance: sv.importance,
                  isProxy: sv.is_proxy
                }))
                .reverse()
            );
          }
        }

        if (mitigateRes.ok) {
          const mitigateData = await mitigateRes.json();

          if (mitigateData.before && mitigateData.after && mitigateData.improvement) {
            const rows = Object.keys(mitigateData.before).map((key) => ({
              metric: METRIC_LABELS[key] || key,
              before: String(mitigateData.before[key]),
              after: String(mitigateData.after[key]),
              diff: `+${mitigateData.improvement[key]}%`,
            }));
            setMitigationData(rows);
          }
        }
      } catch (e) {
        // Silent fallback to hardcoded values
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getMetricColor = (value: number) => {
    if (value < 0.7) return "text-red-600";
    if (value < 0.8) return "text-amber-600";
    return "text-green-600";
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.width;
    let yPos = 20;

    const centerText = (text: string, y: number) => {
      const textWidth = doc.getTextWidth(text);
      doc.text(text, (pageWidth - textWidth) / 2, y);
    };

    // Header
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    centerText("TrialAI", yPos);
    yPos += 10;
    
    doc.setFontSize(16);
    centerText("Official AI Bias Audit Report", yPos);
    yPos += 10;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    centerText(`Trial ID: ${trialId} | Date: ${new Date().toLocaleDateString()}`, yPos);
    yPos += 20;

    // Verdict Summary
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    centerText("VERDICT: GUILTY", yPos);
    yPos += 10;
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    centerText(`Case: ${datasetName} Prediction Model`, yPos);
    yPos += 8;
    centerText("The model has been found guilty of systematic bias.", yPos);
    yPos += 15;

    // Formal Charges Table
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Formal Charges", 14, yPos);
    yPos += 5;

    const chargesData = [
      ["Demographic Parity Violation", fairnessMetrics.demographic_parity < 0.7 ? 'HIGH' : 'MODERATE', "Outcomes are disproportionate across demographic groups."],
      ["Disparate Impact Violation", fairnessMetrics.disparate_impact < 0.7 ? 'HIGH' : 'MODERATE', "Ratio of outcomes falls below legal threshold."],
      ["Equal Opportunity Violation", fairnessMetrics.equal_opportunity < 0.7 ? 'HIGH' : 'MODERATE', "Discrepancy in true positive rates between groups."]
    ];

    autoTable(doc, {
      startY: yPos,
      head: [["Charge", "Severity", "Description"]],
      body: chargesData,
      theme: "plain",
      styles: { lineColor: [0, 0, 0], lineWidth: 0.1 },
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: 'bold' }
    });
    
    yPos = (doc as any).lastAutoTable.finalY + 15;

    // Fairness Metrics Table
    doc.setFont("helvetica", "bold");
    doc.text("Fairness Metrics", 14, yPos);
    yPos += 5;

    const metricsData = [
      ["Demographic Parity", fairnessMetrics.demographic_parity.toString(), "> 0.80", fairnessMetrics.demographic_parity < 0.8 ? "FAIL" : "PASS"],
      ["Equal Opportunity", fairnessMetrics.equal_opportunity.toString(), "> 0.80", fairnessMetrics.equal_opportunity < 0.8 ? "FAIL" : "PASS"],
      ["Disparate Impact", fairnessMetrics.disparate_impact.toString(), "> 0.80", fairnessMetrics.disparate_impact < 0.8 ? "FAIL" : "PASS"]
    ];

    autoTable(doc, {
      startY: yPos,
      head: [["Metric", "Score", "Threshold", "Status"]],
      body: metricsData,
      theme: "plain",
      styles: { lineColor: [0, 0, 0], lineWidth: 0.1 },
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: 'bold' }
    });

    yPos = (doc as any).lastAutoTable.finalY + 15;

    // Top Features Table
    if (yPos > 250) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFont("helvetica", "bold");
    doc.text("Top Features Influence (SHAP)", 14, yPos);
    yPos += 5;

    const shapRows = shapData.map(s => [
      s.feature.replace(" (Proxy)", ""),
      s.importance.toFixed(3),
      s.feature.includes("(Proxy)") || (s as any).isProxy ? "YES" : "NO"
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [["Feature Name", "Importance Score", "Proxy Flag"]],
      body: shapRows,
      theme: "plain",
      styles: { lineColor: [0, 0, 0], lineWidth: 0.1 },
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: 'bold' }
    });

    yPos = (doc as any).lastAutoTable.finalY + 15;

    // Jury Outcome
    if (yPos > 250) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFont("helvetica", "bold");
    doc.text("Jury Outcome Summary", 14, yPos);
    yPos += 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    const deniedCount = JURY_PERSONAS.filter(p => p.outcome === "Denied").length;
    doc.text(`${deniedCount} of ${JURY_PERSONAS.length} personas adversely affected by model bias.`, 14, yPos);
    yPos += 5;

    const juryRows = JURY_PERSONAS.map(p => [
      p.name,
      p.demographic,
      p.occupation,
      p.outcome
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [["Name", "Demographic", "Occupation", "Outcome"]],
      body: juryRows,
      theme: "plain",
      styles: { lineColor: [0, 0, 0], lineWidth: 0.1 },
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: 'bold' }
    });

    yPos = (doc as any).lastAutoTable.finalY + 15;

    // Court Reform Order
    if (yPos > 220) {
      doc.addPage();
      yPos = 20;
    }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Court Reform Order", 14, yPos);
    yPos += 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    doc.text("1. Apply demographic reweighting to training data.", 14, yPos);
    yPos += 6;
    doc.text("2. Remove proxy features heavily correlated with protected classes.", 14, yPos);
    yPos += 6;
    doc.text("3. Apply equalized odds threshold correction.", 14, yPos);
    yPos += 15;

    // Before vs After Mitigation Table
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Impact of Mitigation", 14, yPos);
    yPos += 5;

    const mitigationRows = mitigationData.map(m => [
      m.metric,
      m.before,
      m.after,
      m.diff
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [["Metric", "Before", "After", "Improvement %"]],
      body: mitigationRows,
      theme: "plain",
      styles: { lineColor: [0, 0, 0], lineWidth: 0.1 },
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: 'bold' }
    });

    // Footer
    const pageCount = (doc as any).internal.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFont("helvetica", "italic");
      doc.setFontSize(8);
      centerText(`Generated by TrialAI — AI Fairness Audit Platform | ${new Date().toLocaleString()}`, doc.internal.pageSize.height - 10);
    }

    const dateStr = new Date().toISOString().split('T')[0];
    const safeDatasetName = datasetName.replace(/[^a-zA-Z0-9]/g, "-");
    doc.save(`TrialAI-Report-${safeDatasetName}-${dateStr}.pdf`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-foreground/50" />
          <p className="text-foreground/60 font-medium">Loading verdict data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground pb-24 font-sans selection:bg-red-500/20">
      
      {/* Demo Banner */}
      <div className="bg-amber-100 border-b border-amber-200 py-2.5 px-4 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto flex items-center justify-center gap-3">
          <span className="bg-amber-500 text-amber-950 text-xs font-bold px-2 py-0.5 rounded shadow-sm">Demo Mode</span>
          <p className="text-sm font-medium text-amber-900">
            COMPAS Recidivism Dataset — A real-world AI bias case from 2016
          </p>
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-6 pt-16">
        <motion.div
          initial="hidden"
          animate="show"
          viewport={{ once: true }}
          variants={{
            hidden: {},
            show: {
              transition: {
                staggerChildren: 0.15,
              },
            },
          }}
          className="space-y-16"
        >
          {/* 1. Verdict Header */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS} className="text-center space-y-4">
            <p className="text-sm font-mono text-foreground/50 font-bold uppercase tracking-widest">
              Trial #{trialId} — {datasetName} Prediction
            </p>
            <div className="inline-flex items-center gap-3 px-6 py-3 bg-red-100 text-red-700 border-2 border-red-300 rounded-xl shadow-sm">
              <Gavel className="w-8 h-8" />
              <h1 className="text-4xl font-black tracking-tight">GUILTY</h1>
            </div>
            <p className="text-xl text-foreground/70 font-medium">
              The model has been found guilty of systematic bias.
            </p>
          </motion.section>

          {/* 2. Charges */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS}>
            <h2 className="text-2xl font-bold mb-4 border-b border-border pb-2">Formal Charges</h2>
            <div className="bg-surface border border-border rounded-xl p-2 space-y-2">
              
              <div className="p-4 bg-background border border-border rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-semibold text-lg flex items-center gap-2">
                    Charge 1: Demographic Parity Violation
                  </h3>
                  <p className="text-foreground/60 text-sm"> African Americans are 38% less likely to receive a favorable outcome.</p>
                </div>
                <div className={`shrink-0 inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-bold text-xs ${fairnessMetrics.demographic_parity < 0.7 ? 'bg-red-100 text-red-700 border-red-200' : 'bg-amber-100 text-amber-700 border-amber-200'}`}>
                  <AlertTriangle className="w-3.5 h-3.5" /> Severity: {fairnessMetrics.demographic_parity < 0.7 ? 'HIGH' : 'MODERATE'}
                </div>
              </div>

              <div className="p-4 bg-background border border-border rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-semibold text-lg flex items-center gap-2">
                    Charge 2: Disparate Impact Violation
                  </h3>
                  <p className="text-foreground/60 text-sm">Ratio of outcomes between minority and majority groups falls below 80% legal threshold.</p>
                </div>
                <div className={`shrink-0 inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-bold text-xs ${fairnessMetrics.disparate_impact < 0.7 ? 'bg-red-100 text-red-700 border-red-200' : 'bg-amber-100 text-amber-700 border-amber-200'}`}>
                  <AlertTriangle className="w-3.5 h-3.5" /> Severity: {fairnessMetrics.disparate_impact < 0.7 ? 'HIGH' : 'MODERATE'}
                </div>
              </div>

              <div className="p-4 bg-background border border-border rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-semibold text-lg flex items-center gap-2">
                    Charge 3: Equal Opportunity Violation
                  </h3>
                  <p className="text-foreground/60 text-sm">Discrepancy in true positive rates between protected groups.</p>
                </div>
                <div className={`shrink-0 inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-bold text-xs ${fairnessMetrics.equal_opportunity < 0.7 ? 'bg-red-100 text-red-700 border-red-200' : 'bg-amber-100 text-amber-700 border-amber-200'}`}>
                  <AlertTriangle className="w-3.5 h-3.5" /> Severity: {fairnessMetrics.equal_opportunity < 0.7 ? 'HIGH' : 'MODERATE'}
                </div>
              </div>

            </div>
          </motion.section>

          {/* 3. Jury Outcome Summary */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS}>
            <div className="flex items-center justify-between mb-4 border-b border-border pb-2">
              <h2 className="text-2xl font-bold">Jury Outcome Summary</h2>
            </div>
            <p className="text-lg font-medium text-red-600 bg-red-50 p-4 rounded-lg border border-red-100 mb-6 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              {JURY_PERSONAS.filter(p => p.outcome === "Denied").length} of {JURY_PERSONAS.length} jury members were adversely affected by model bias
            </p>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {JURY_PERSONAS.map(persona => {
                const isDenied = persona.outcome === "Denied";
                return (
                  <div key={persona.id} className={`p-3 rounded-xl border flex items-center gap-3 ${isDenied ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold shrink-0 ${isDenied ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                      {persona.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className="overflow-hidden">
                      <p className="font-semibold text-sm text-foreground truncate">{persona.name}</p>
                      <p className="text-xs text-foreground/60 truncate">{persona.demographic}</p>
                    </div>
                    <div className="ml-auto shrink-0">
                      {isDenied ? <XCircle className="w-5 h-5 text-red-500" /> : <CheckCircle2 className="w-5 h-5 text-green-500" />}
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.section>

          {/* 4. Evidence Summary */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS}>
            <h2 className="text-2xl font-bold mb-4 border-b border-border pb-2">Evidence Summary</h2>
            <div className="grid md:grid-cols-2 gap-6">
              
              {/* Left: SHAP Chart */}
              <div className="bg-surface border border-border rounded-xl p-5">
                <h3 className="font-semibold mb-1">Key Influencing Features</h3>
                <p className="text-xs text-foreground/50 mb-6">SHAP global importance analysis</p>
                <div className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                      <XAxis type="number" hide />
                      <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#64748B" }} width={120} />
                      <Tooltip cursor={{ fill: 'rgba(0,0,0,0.05)' }} contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                      <Bar dataKey="importance" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={20}>
                        {
                          shapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.feature.includes("Proxy") || (entry as any).isProxy ? "#EF4444" : "#3B82F6"} />
                          ))
                        }
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Right: Metrics */}
              <div className="space-y-4">
                <div className="p-4 bg-surface border border-border rounded-xl flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-sm">Demographic Parity</h3>
                    <p className="text-xs text-foreground/60">Target: &gt;0.80</p>
                  </div>
                  <div className={`text-2xl font-bold ${getMetricColor(fairnessMetrics.demographic_parity)}`}>{fairnessMetrics.demographic_parity}</div>
                </div>
                <div className="p-4 bg-surface border border-border rounded-xl flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-sm">Disparate Impact</h3>
                    <p className="text-xs text-foreground/60">Target: &gt;0.80</p>
                  </div>
                  <div className={`text-2xl font-bold ${getMetricColor(fairnessMetrics.disparate_impact)}`}>{fairnessMetrics.disparate_impact}</div>
                </div>
                <div className="p-4 bg-surface border border-border rounded-xl flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-sm">Equal Opportunity</h3>
                    <p className="text-xs text-foreground/60">Target: &gt;0.80</p>
                  </div>
                  <div className={`text-2xl font-bold ${getMetricColor(fairnessMetrics.equal_opportunity)}`}>{fairnessMetrics.equal_opportunity}</div>
                </div>
              </div>

            </div>
          </motion.section>

          {/* 5. Reform Order */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS}>
            <div className="bg-background border-2 border-border shadow-sm rounded-xl p-8 font-serif relative overflow-hidden">
              <div className="absolute top-0 left-0 w-1 h-full bg-gold" />
              <div className="text-center mb-8 border-b-2 border-border pb-6">
                <Gavel className="w-8 h-8 mx-auto text-gold mb-2" />
                <h2 className="text-2xl font-bold uppercase tracking-widest text-foreground">Court Reform Order #{trialId}</h2>
              </div>
              <p className="text-foreground/80 mb-6 leading-relaxed">
                Pursuant to the findings of algorithmic bias, the Defendant model is hereby ordered to undergo immediate algorithmic mitigation. The following mandatory steps must be executed before deployment:
              </p>
              <ul className="space-y-4 font-sans">
                <li className="flex items-start gap-3">
                  <div className="mt-1 w-6 h-6 rounded-full bg-gold/20 text-gold flex items-center justify-center font-bold text-xs shrink-0">1</div>
                  <div>
                    <p className="font-bold">Apply demographic reweighting to training data.</p>
                    <p className="text-sm text-foreground/60">Balance class weights to ensure equal representation of minority groups.</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="mt-1 w-6 h-6 rounded-full bg-gold/20 text-gold flex items-center justify-center font-bold text-xs shrink-0">2</div>
                  <div>
                    <p className="font-bold">Remove proxy features heavily correlated with protected classes.</p>
                    <p className="text-sm text-foreground/60">Feature drop identified attributes heavily correlated with race.</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <div className="mt-1 w-6 h-6 rounded-full bg-gold/20 text-gold flex items-center justify-center font-bold text-xs shrink-0">3</div>
                  <div>
                    <p className="font-bold">Apply equalized odds threshold correction.</p>
                    <p className="text-sm text-foreground/60">Post-processing adjustments to align true positive rates across groups.</p>
                  </div>
                </li>
              </ul>
            </div>
          </motion.section>

          {/* 6. Before vs After */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS}>
            <h2 className="text-2xl font-bold mb-4 border-b border-border pb-2">Impact of Mitigation</h2>
            <div className="border border-border rounded-xl overflow-hidden">
              <table className="w-full text-left text-sm">
                <thead className="bg-surface/50 border-b border-border uppercase text-xs text-foreground/50">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Metric</th>
                    <th className="px-6 py-4 font-semibold">Original Score</th>
                    <th className="px-6 py-4 font-semibold">After Mitigation</th>
                    <th className="px-6 py-4 font-semibold text-right">Improvement</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border bg-background">
                  {mitigationData.map((row) => (
                    <tr key={row.metric} className="hover:bg-surface/30">
                      <td className="px-6 py-4 font-medium">{row.metric}</td>
                      <td className="px-6 py-4 font-mono text-red-600 font-medium">{row.before}</td>
                      <td className="px-6 py-4 font-mono text-green-600 font-medium">{row.after}</td>
                      <td className="px-6 py-4 font-mono text-green-600 font-bold text-right">{row.diff}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.section>

          {/* 7. Retrial Result & 8. Actions */}
          <motion.section variants={FADE_UP_ANIMATION_VARIANTS} className="space-y-8">
            <div className="bg-green-50 border border-green-200 text-green-800 p-6 rounded-xl flex items-center justify-center gap-3">
              <CheckCircle2 className="w-6 h-6" />
              <h3 className="text-lg font-bold">Retrial Passed — Model meets minimum fairness thresholds</h3>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <button onClick={generatePDF} className="w-full sm:w-auto px-8 py-4 bg-foreground text-background font-semibold rounded-lg flex items-center justify-center gap-2 hover:bg-foreground/90 transition-all shadow-md">
                <Download className="w-5 h-5" />
                Download Report (PDF)
              </button>
              <Link href="/trial/upload" className="w-full sm:w-auto px-8 py-4 bg-surface border border-border text-foreground font-semibold rounded-lg flex items-center justify-center gap-2 hover:bg-surface/80 transition-all">
                <PlayCircle className="w-5 h-5" />
                Start Your Own Trial
              </Link>
            </div>
          </motion.section>

        </motion.div>
      </main>
    </div>
  );
}
