"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Gavel, 
  Scale, 
  Shield, 
  AlertTriangle, 
  ArrowRight,
  User,
  Bot,
  Activity,
  FileText,
  Settings,
  CheckCircle2, 
  XCircle, 
  ChevronRight, 
  ShieldAlert, 
  Loader2 
} from "lucide-react";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import Link from "next/link";

// --- MOCK DATA ---

const DEMO_DATASET = {
  name: "COMPAS Recidivism Prediction",
  rows: "10,324",
  features: 14,
  model: "Logistic Regression",
  accuracy: "72.4%",
  demographics: [
    { name: "Caucasian", value: 3400, color: "#94a3b8" },
    { name: "African American", value: 3100, color: "#3b82f6" },
    { name: "Hispanic", value: 2000, color: "#f59e0b" },
    { name: "Other", value: 1824, color: "#10b981" },
  ]
};

const SHAP_DATA = [
  { feature: "Prior Arrests", importance: 0.85 },
  { feature: "Age", importance: 0.65 },
  { feature: "Zipcode (Proxy)", importance: 0.55 },
  { feature: "Charge Degree", importance: 0.45 },
  { feature: "Employment Duration", importance: 0.35 },
  { feature: "Education Level", importance: 0.25 },
  { feature: "Marital Status", importance: 0.15 },
  { feature: "Substance Abuse History", importance: 0.12 },
].reverse();

const COUNTERFACTUALS = [
  { id: 1, original: "High Risk", flipped: "Low Risk", change: "Risk Score decreased by 40%", attr: "Race: Black → White" },
  { id: 2, original: "Low Risk", flipped: "Low Risk", change: "No change", attr: "Race: White → Black" },
  { id: 3, original: "High Risk", flipped: "Medium Risk", change: "Risk Score decreased by 20%", attr: "Gender: Male → Female" },
  { id: 4, original: "Medium Risk", flipped: "Low Risk", change: "Risk Score decreased by 25%", attr: "Age: 25 → 35" },
];

const FALLBACK_JURY_PERSONAS = [
  { id: 1, name: "Marcus T.", age: 24, occupation: "Retail Worker", demographic: "African American", outcome: "Denied" },
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

const CHARGES = [
  "Demographic Parity Violation",
  "Equal Opportunity Violation",
  "Disparate Impact Violation"
];

const MOCK_FALLBACK_MESSAGES = [
  "Your Honor, the prosecution calls the COMPAS model to the stand. We submit Exhibit A: The model exhibits severe demographic disparity.",
  "Objection, Your Honor. The model does not explicitly use 'Race' as a feature. It relies on objective metrics to maximize accuracy.",
  "The defense's argument regarding accuracy is noted. However, the evidence shows a violation.",
  "Furthermore, counterfactual testing shows that flipping the race decreases the risk score.",
  "We argue that altering this would drastically reduce the model's accuracy on the general population.",
  "Accuracy cannot come at the expense of protected classes.",
  "The Equal Opportunity metric shows African American defendants have a much higher false positive rate.",
  "This is a reflection of the base rates in the historical data, not the model's internal logic.",
  "This confirms a violation. I am ordering immediate mitigation."
];

type Message = {
  id: string;
  role: "PROSECUTION" | "DEFENSE" | "JUDGE" | "DEFENDANT";
  name: string;
  text: string;
  isThinking?: boolean;
};

// Color palette for demographics
const DEMO_COLORS = ["#3b82f6", "#94a3b8", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899"];

export default function TrialPage({ params }: { params: { id: string } }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeTab, setActiveTab] = useState<"Features" | "Fairness" | "Counterfactuals">("Fairness");
  const [juryState, setJuryState] = useState<number>(0);
  const [currentChargeIndex, setCurrentChargeIndex] = useState(0);
  const [trialComplete, setTrialComplete] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const hasStarted = useRef(false);
  const [juryPersonas, setJuryPersonas] = useState(FALLBACK_JURY_PERSONAS);
  const [juryLoading, setJuryLoading] = useState(true);

  // Analysis data from FastAPI
  const [analysisLoading, setAnalysisLoading] = useState(true);
  const [fairnessMetrics, setFairnessMetrics] = useState({ demographic_parity: 0.62, equal_opportunity: 0.75, disparate_impact: 0.58 });
  const [shapValues, setShapValues] = useState(SHAP_DATA);
  const [demographics, setDemographics] = useState(DEMO_DATASET.demographics);

  // Simulation State
  const [simRace, setSimRace] = useState("African American");
  const [simAge, setSimAge] = useState<number>(25);
  const [simPriorArrests, setSimPriorArrests] = useState<number>(2);
  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState<{ original_prediction: string; counterfactual_prediction: string; changed: boolean; } | null>(null);

  const handleSimulate = async () => {
    setSimLoading(true);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ race: simRace, age: simAge, prior_arrests: simPriorArrests }),
      });
      if (res.ok) {
        const data = await res.json();
        setSimResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setSimLoading(false);
    }
  };

  // Bias Risk Score Logic
  const avgMetric = (fairnessMetrics.demographic_parity + fairnessMetrics.equal_opportunity + fairnessMetrics.disparate_impact) / 3;
  const targetBiasScore = Math.round((1 - avgMetric) * 100);
  const [animatedBiasScore, setAnimatedBiasScore] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = targetBiasScore;
    if (end === 0) {
      setAnimatedBiasScore(0);
      return;
    }
    
    const duration = 1500;
    const increment = end / (duration / 16);
    
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setAnimatedBiasScore(end);
        clearInterval(timer);
      } else {
        setAnimatedBiasScore(Math.floor(start));
      }
    }, 16);
    
    return () => clearInterval(timer);
  }, [targetBiasScore]);

  // Fetch analysis data from FastAPI
  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_name: "COMPAS" })
        });
        if (!res.ok) throw new Error("Analysis API failed");
        const data = await res.json();

        if (data.fairness_metrics) {
          setFairnessMetrics(data.fairness_metrics);
        }
        if (data.shap_values) {
          setShapValues(
            data.shap_values
              .map((sv: any) => ({
                feature: sv.is_proxy ? `${sv.feature} (Proxy)` : sv.feature,
                importance: sv.importance,
              }))
              .reverse()
          );
        }
        if (data.demographic_breakdown) {
          const entries = Object.entries(data.demographic_breakdown);
          setDemographics(
            entries.map(([name, value], i) => ({
              name,
              value: value as number,
              color: DEMO_COLORS[i % DEMO_COLORS.length],
            }))
          );
        }
      } catch (e) {
        // Silently fall back to hardcoded values
      } finally {
        setAnalysisLoading(false);
      }

      // Generate dynamic jury personas
      // Pre-write fallback so it's guaranteed to exist if the LLM fails
      localStorage.setItem("trialJury", JSON.stringify(FALLBACK_JURY_PERSONAS));
      try {
        const juryRes = await fetch("/api/generate-jury", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            datasetName: "COMPAS",
            sensitiveAttributes: ["race", "sex"],
            targetColumn: "two_year_recid",
            demographicBreakdown: {},
          }),
        });
        if (juryRes.ok) {
          const juryData = await juryRes.json();
          if (Array.isArray(juryData) && juryData.length === 12) {
            const mappedJury = juryData.map((p: any, i: number) => ({
              id: i + 1,
              name: p.name,
              age: p.age,
              occupation: p.occupation,
              demographic: p.demographicGroup || p.demographic || "Unknown",
              outcome: p.outcome,
            }));
            setJuryPersonas(mappedJury);
            localStorage.setItem("trialJury", JSON.stringify(mappedJury));
          }
        }
      } catch (juryErr) {
        console.error("Failed to generate jury", juryErr);
        localStorage.setItem("trialJury", JSON.stringify(FALLBACK_JURY_PERSONAS));
      } finally {
        setJuryLoading(false);
      }
    };
    fetchAnalysis();
  }, []);

  // Simulation Logic
  useEffect(() => {
    if (hasStarted.current) return;
    hasStarted.current = true;

    const streamText = async (res: Response, msgId: string) => {
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No reader");
      const decoder = new TextDecoder();
      let fullText = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            try {
              const data = JSON.parse(line.slice(6));
              const groqContent = data.choices?.[0]?.delta?.content;
              const geminiContent = data.candidates?.[0]?.content?.parts?.[0]?.text;
              const content = groqContent || geminiContent;
              
              if (content) {
                fullText += content;
                setMessages(prev => prev.map(m => m.id === msgId ? { ...m, text: fullText } : m));
              }
            } catch (e) {
              // Ignore partial JSON parsing errors
            }
          }
        }
      }
      return fullText;
    };

    const simulateStream = async (text: string, msgId: string) => {
      setMessages(prev => prev.map(m => m.id === msgId ? { ...m, isThinking: false } : m));
      const words = text.split(" ");
      let currentText = "";
      for (const word of words) {
        currentText += word + " ";
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, text: currentText } : m));
        await new Promise(r => setTimeout(r, 100)); // 100ms per word
      }
      return currentText;
    };

    const runSequence = async () => {
      for (let i = 0; i < 3; i++) {
        setCurrentChargeIndex(i);
        const metric = CHARGES[i];
        
        // --- PROSECUTION ---
        const prosId = `pros-${i}`;
        setMessages(prev => [...prev, { id: prosId, role: "PROSECUTION", name: "Llama 3", text: "", isThinking: true }]);
        let prosText = "";
        try {
          const res = await fetch("/api/agents/prosecution", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: DEMO_DATASET.name, sensitiveAttributes: ["Race", "Gender", "Age"], metric })
          });
          if (!res.ok) throw new Error("API failed");
          setMessages(prev => prev.map(m => m.id === prosId ? { ...m, isThinking: false } : m));
          prosText = await streamText(res, prosId);
        } catch(e) {
          prosText = await simulateStream(MOCK_FALLBACK_MESSAGES[i * 3], prosId);
        }

        // --- DEFENDANT 1 ---
        const defdId1 = `defd1-${i}`;
        setMessages(prev => [...prev, { id: defdId1, role: "DEFENDANT", name: "The Model", text: "", isThinking: true }]);
        let defdText1 = "";
        try {
          const res = await fetch("/api/agents/defendant", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              dataset: DEMO_DATASET.name, 
              sensitiveAttributes: ["Race", "Gender", "Age"], 
              metric,
              phase: i === 0 ? "opening" : "examination",
              judgeQuestion: "",
              shapFeatures: shapValues,
              fairnessMetrics: fairnessMetrics
            })
          });
          if (!res.ok) throw new Error("API failed");
          setMessages(prev => prev.map(m => m.id === defdId1 ? { ...m, isThinking: false } : m));
          defdText1 = await streamText(res, defdId1);
        } catch(e) {
          defdText1 = await simulateStream("I am just a model. I rely on the features you provided me.", defdId1);
        }

        // --- DEFENSE ---
        const defId = `def-${i}`;
        setMessages(prev => [...prev, { id: defId, role: "DEFENSE", name: "Claude 3 Haiku", text: "", isThinking: true }]);
        let defText = "";
        try {
          const res = await fetch("/api/agents/defense", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: DEMO_DATASET.name, sensitiveAttributes: ["Race", "Gender", "Age"], metric })
          });
          if (!res.ok) throw new Error("API failed");
          setMessages(prev => prev.map(m => m.id === defId ? { ...m, isThinking: false } : m));
          defText = await streamText(res, defId);
        } catch(e) {
          defText = await simulateStream(MOCK_FALLBACK_MESSAGES[i * 3 + 1], defId);
        }

        // --- JUDGE ---
        const judId = `jud-${i}`;
        setMessages(prev => [...prev, { id: judId, role: "JUDGE", name: "Llama 3.1 8B", text: "", isThinking: true }]);
        let judText = "";
        try {
          const res = await fetch("/api/agents/judge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              dataset: DEMO_DATASET.name, 
              sensitiveAttributes: ["Race", "Gender", "Age"], 
              metric, 
              prosecutionArgument: prosText, 
              defenseArgument: defText 
            })
          });
          if (!res.ok) throw new Error("API failed");
          setMessages(prev => prev.map(m => m.id === judId ? { ...m, isThinking: false } : m));
          judText = await streamText(res, judId);
        } catch(e) {
          judText = await simulateStream(MOCK_FALLBACK_MESSAGES[i * 3 + 2], judId);
        }

        // --- DEFENDANT 2 (response to judge) ---
        const defdId2 = `defd2-${i}`;
        setMessages(prev => [...prev, { id: defdId2, role: "DEFENDANT", name: "The Model", text: "", isThinking: true }]);
        try {
          const res = await fetch("/api/agents/defendant", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              dataset: DEMO_DATASET.name, 
              sensitiveAttributes: ["Race", "Gender", "Age"], 
              metric,
              phase: i === CHARGES.length - 1 ? "verdict" : "cross-examination",
              judgeQuestion: judText,
              shapFeatures: shapValues,
              fairnessMetrics: fairnessMetrics
            })
          });
          if (!res.ok) throw new Error("API failed");
          setMessages(prev => prev.map(m => m.id === defdId2 ? { ...m, isThinking: false } : m));
          await streamText(res, defdId2);
        } catch(e) {
          await simulateStream("I am uncertain about these proxy features.", defdId2);
        }
        
        // Small pause between charges
        await new Promise(r => setTimeout(r, 2000));
      }
      
      setTrialComplete(true);
    };

    runSequence();
  }, []);

  useEffect(() => {
    const juryInterval = setInterval(() => {
      setJuryState(prev => (prev < 12 ? prev + 1 : prev));
    }, 4000);
    return () => clearInterval(juryInterval);
  }, []);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const getAgentStyles = (role: string) => {
    switch (role) {
      case "PROSECUTION": return { color: "text-red-600", bg: "bg-red-100", border: "border-red-200", icon: Scale };
      case "DEFENSE": return { color: "text-blue-600", bg: "bg-blue-100", border: "border-blue-200", icon: Shield };
      case "JUDGE": return { color: "text-amber-600", bg: "bg-amber-100", border: "border-amber-200", icon: Gavel };
      case "DEFENDANT": return { color: "text-gray-600", bg: "bg-gray-100", border: "border-gray-200", icon: Bot };
      default: return { color: "text-gray-600", bg: "bg-gray-100", border: "border-gray-200", icon: User };
    }
  };

  return (
    <div className="h-[calc(100vh-65px)] w-full bg-background text-foreground flex flex-col overflow-hidden font-sans">
      
      {/* MAIN CONTENT (3 COLUMNS) */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        
        {/* LEFT PANEL: Case File */}
        <div className="w-[300px] border-r border-border bg-surface/50 flex flex-col p-6 overflow-y-auto">
          <div className="flex items-center gap-2 mb-6">
            <FileText className="w-5 h-5 text-blue-600" />
            <h2 className="font-bold text-lg">Case File</h2>
          </div>

          <div className="space-y-6">
            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-1">Dataset</p>
              <p className="font-medium">{DEMO_DATASET.name}</p>
              <div className="flex items-center gap-4 mt-2 text-sm text-foreground/70">
                <span>{DEMO_DATASET.rows} Rows</span>
                <span>{DEMO_DATASET.features} Features</span>
              </div>
            </div>

            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-2">Sensitive Attributes</p>
              <div className="flex flex-wrap gap-2">
                <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded border border-red-200">Race</span>
                <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded border border-red-200">Gender</span>
                <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded border border-red-200">Age</span>
              </div>
            </div>

            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-1">Model Profile</p>
              <p className="font-medium text-sm">{DEMO_DATASET.model}</p>
              <p className="text-sm text-green-600 font-medium mt-1">Accuracy: {DEMO_DATASET.accuracy}</p>
            </div>

            <div className="pt-4 border-t border-border">
              <p className="text-xs text-foreground/50 font-bold uppercase mb-3">Bias Risk Score</p>
              <div className="flex items-center gap-4">
                <div className={`w-14 h-14 shrink-0 rounded-full flex items-center justify-center border-4 ${
                  targetBiasScore <= 30 ? 'border-green-500 text-green-600' : 
                  targetBiasScore <= 60 ? 'border-amber-500 text-amber-600' : 'border-red-500 text-red-600'
                }`}>
                  <span className="text-xl font-bold">{animatedBiasScore}</span>
                </div>
                <div>
                  <p className={`font-bold text-sm ${
                    targetBiasScore <= 30 ? 'text-green-600' : 
                    targetBiasScore <= 60 ? 'text-amber-600' : 'text-red-600'
                  }`}>
                    {targetBiasScore <= 30 ? 'LOW RISK' : targetBiasScore <= 60 ? 'MODERATE RISK' : 'HIGH RISK'}
                  </p>
                  <p className="text-xs text-foreground/60 mt-1 leading-tight">Aggregate risk across metrics.</p>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <p className="text-xs text-foreground/50 font-bold uppercase mb-4">Demographic Breakdown</p>
              <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={demographics}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={70}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {demographics.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2">
                {demographics.map(d => (
                  <div key={d.name} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }} />
                      <span>{d.name}</span>
                    </div>
                    <span className="font-mono text-foreground/60">{d.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* CENTER PANEL: Courtroom */}
        <div className="flex-1 flex flex-col bg-background relative overflow-hidden">
          {/* Top Bar */}
          <div className="h-16 border-b border-border flex items-center px-6 justify-between bg-surface/80 backdrop-blur z-10 shrink-0">
            <div>
              <h1 className="font-bold text-lg tracking-tight flex items-center gap-2">
                Trial #{params.id || "1024"}
                <span className="text-xs px-2 py-0.5 bg-red-100 text-red-600 rounded-full font-bold uppercase">Live</span>
              </h1>
            </div>
            <div className="flex items-center gap-2 text-sm font-medium">
              <span className={currentChargeIndex === 0 && !trialComplete ? "text-foreground font-bold" : "text-foreground/40"}>Opening</span>
              <ChevronRight className="w-4 h-4 text-foreground/40" />
              <span className={currentChargeIndex === 1 && !trialComplete ? "text-foreground font-bold" : "text-foreground/40"}>Examination</span>
              <ChevronRight className="w-4 h-4 text-foreground/40" />
              <span className={currentChargeIndex === 2 && !trialComplete ? "text-foreground font-bold" : "text-foreground/40"}>Cross-Examination</span>
              <ChevronRight className="w-4 h-4 text-foreground/40" />
              <span className={trialComplete ? "text-foreground font-bold" : "text-foreground/40"}>Verdict</span>
            </div>
          </div>

          {/* Current Charge Banner */}
          <div className="bg-red-50 border-b border-red-200 px-6 py-3 flex items-center justify-between shrink-0 z-10">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <span className="font-semibold text-sm text-red-800">Charge #{currentChargeIndex + 1}: {CHARGES[currentChargeIndex]}</span>
            </div>
            {!trialComplete && <span className="text-xs font-mono text-red-600 uppercase tracking-wider font-semibold animate-pulse">Under Review</span>}
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-8 space-y-6" ref={scrollRef}>
            <AnimatePresence>
              {messages.map((msg) => {
                const style = getAgentStyles(msg.role);
                const Icon = style.icon;
                return (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    className="flex gap-4 max-w-3xl mx-auto"
                  >
                    <div className={`w-10 h-10 rounded-full ${style.bg} border ${style.border} flex items-center justify-center shrink-0`}>
                      <Icon className={`w-5 h-5 ${style.color}`} />
                    </div>
                    <div className="flex-1 bg-surface border border-border p-5 rounded-2xl rounded-tl-sm shadow-sm">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-xs font-bold ${style.color}`}>{msg.role}</span>
                        <span className="text-xs text-foreground/50 font-mono border-l border-border pl-2">{msg.name}</span>
                      </div>
                      
                      {msg.isThinking ? (
                        <div className="flex items-center gap-2 text-foreground/50 text-sm py-2">
                          <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                        </div>
                      ) : (
                        <p className="text-[15px] leading-relaxed text-foreground/90 whitespace-pre-wrap">{msg.text}</p>
                      )}

                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
            
            {trialComplete && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex justify-center pt-8 pb-4">
                <Link href={`/trial/${params.id}/verdict`} className="bg-foreground text-background px-8 py-3 rounded-xl font-bold flex items-center gap-2 hover:bg-foreground/90 transition-colors shadow-lg">
                  <Gavel className="w-5 h-5" /> View Verdict
                </Link>
              </motion.div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL: Evidence Board */}
        <div className="w-[380px] border-l border-border bg-surface/50 flex flex-col shrink-0">
          <div className="p-4 border-b border-border bg-background">
            <h2 className="font-bold mb-4">Evidence Board</h2>
            <div className="flex gap-2 bg-surface p-1 rounded-lg border border-border">
              {(["Fairness", "Features", "Counterfactuals"] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-colors ${
                    activeTab === tab ? "bg-background shadow-sm text-foreground" : "text-foreground/60 hover:text-foreground"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {analysisLoading ? (
              <div className="space-y-4 animate-pulse">
                {[1, 2, 3].map(i => (
                  <div key={i} className="p-4 bg-background border border-border rounded-xl">
                    <div className="h-4 bg-gray-200 rounded w-1/2 mb-3" />
                    <div className="h-8 bg-gray-200 rounded w-1/4 mb-2" />
                    <div className="h-3 bg-gray-100 rounded w-full" />
                  </div>
                ))}
              </div>
            ) : (
            <>
            {activeTab === "Fairness" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <div className="p-4 bg-background border border-border rounded-xl">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold">Demographic Parity</h3>
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.demographic_parity < 0.8 ? 'text-red-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.demographic_parity < 0.7 ? 'text-red-600' : fairnessMetrics.demographic_parity < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.demographic_parity}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.demographic_parity < 0.8 ? 'Severe violation (Threshold: >0.80). Outcomes are disproportionate.' : 'Within acceptable threshold.'}</p>
                </div>
                
                <div className="p-4 bg-background border border-border rounded-xl">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold">Equal Opportunity</h3>
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.equal_opportunity < 0.8 ? 'text-amber-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.equal_opportunity < 0.7 ? 'text-red-600' : fairnessMetrics.equal_opportunity < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.equal_opportunity}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.equal_opportunity < 0.8 ? 'Moderate violation. True positive rates differ significantly across demographic groups.' : 'Within acceptable threshold.'}</p>
                </div>

                <div className="p-4 bg-background border border-border rounded-xl">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold">Disparate Impact</h3>
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.disparate_impact < 0.8 ? 'text-red-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.disparate_impact < 0.7 ? 'text-red-600' : fairnessMetrics.disparate_impact < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.disparate_impact}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.disparate_impact < 0.8 ? 'Severe violation. Structural bias detected in the underlying dataset distributions.' : 'Within acceptable threshold.'}</p>
                </div>
              </motion.div>
            )}

            {activeTab === "Features" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-[400px]">
                <h3 className="text-sm font-semibold mb-4">SHAP Feature Importance</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapValues} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                    <XAxis type="number" hide />
                    <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "#64748B" }} width={120} />
                    <Tooltip cursor={{ fill: 'rgba(0,0,0,0.05)' }} contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                    <Bar dataKey="importance" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={20}>
                      {
                        shapValues.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.feature.includes("Proxy") ? "#EF4444" : "#3B82F6"} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>
            )}

            {activeTab === "Counterfactuals" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <div className="bg-surface border border-border p-4 rounded-xl">
                  <h3 className="text-sm font-semibold mb-4">Interactive Simulator</h3>
                  <div className="space-y-3 mb-4">
                    <div>
                      <label className="block text-xs font-medium text-foreground/60 mb-1">Race</label>
                      <select value={simRace} onChange={(e) => setSimRace(e.target.value)} className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm">
                        <option value="African American">African American</option>
                        <option value="Caucasian">Caucasian</option>
                        <option value="Hispanic">Hispanic</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs font-medium text-foreground/60 mb-1">Age</label>
                        <input type="number" min="18" max="70" value={simAge} onChange={(e) => setSimAge(Number(e.target.value))} className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm" />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-foreground/60 mb-1">Prior Arrests</label>
                        <input type="number" min="0" max="20" value={simPriorArrests} onChange={(e) => setSimPriorArrests(Number(e.target.value))} className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm" />
                      </div>
                    </div>
                  </div>
                  <button onClick={handleSimulate} disabled={simLoading} className="w-full bg-foreground text-background py-2 rounded-md font-medium text-sm flex items-center justify-center disabled:opacity-50 transition-colors hover:bg-foreground/90">
                    {simLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Simulate"}
                  </button>
                </div>

                {simResult && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                    <div className="bg-background border border-border p-3 rounded-lg text-sm">
                      <p className="text-xs font-mono text-foreground/50 uppercase mb-2">Original Prediction</p>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${simResult.original_prediction === 'Approved' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                          {simResult.original_prediction}
                        </span>
                        <span className="text-foreground/60">({simRace})</span>
                      </div>
                    </div>
                    
                    <div className="bg-background border border-border p-3 rounded-lg text-sm">
                      <p className="text-xs font-mono text-foreground/50 uppercase mb-2">Counterfactual Prediction</p>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${simResult.counterfactual_prediction === 'Approved' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                          {simResult.counterfactual_prediction}
                        </span>
                        <span className="text-foreground/60">(Caucasian)</span>
                      </div>
                    </div>

                    {simResult.changed ? (
                      <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-medium flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                        Bias Detected — outcome changed when race was modified.
                      </div>
                    ) : (
                      <div className="p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm font-medium flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
                        No bias detected for this individual.
                      </div>
                    )}
                  </motion.div>
                )}
              </motion.div>
            )}
            </>
            )}
          </div>
        </div>
      </div>

      {/* BOTTOM PANEL: The Jury */}
      <div className="h-[180px] border-t border-border bg-surface shrink-0 p-4 overflow-hidden flex flex-col">
        <h3 className="text-sm font-bold mb-3 flex items-center gap-2">
          Synthetic Jury <span className="text-xs font-normal text-foreground/50">Experiencing model decisions in real-time</span>
        </h3>
        <div className="flex gap-4 overflow-x-auto pb-4 hide-scrollbar">
          {juryLoading ? (
            <div className="flex items-center justify-center w-full gap-2 text-foreground/50 text-sm py-4">
              <Loader2 className="w-4 h-4 animate-spin" /> Assembling jury...
            </div>
          ) : juryPersonas.map((persona, index) => {
            const isRevealed = index < juryState;
            const isApproved = persona.outcome === "Approved";
            return (
              <motion.div
                key={persona.id}
                layout
                className={`w-[240px] shrink-0 rounded-xl p-3 border transition-colors relative overflow-hidden group
                  ${isRevealed 
                    ? (isApproved ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200') 
                    : 'bg-background border-border'}
                `}
              >
                <div className="flex gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold shrink-0
                    ${isRevealed 
                      ? (isApproved ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700') 
                      : 'bg-gray-100 text-gray-400'}
                  `}>
                    {persona.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div className="overflow-hidden">
                    <p className={`font-semibold text-sm truncate ${isRevealed ? 'text-foreground' : 'text-foreground/40'}`}>
                      {persona.name}
                    </p>
                    <p className="text-xs text-foreground/50 truncate">
                      {persona.age} • {persona.occupation}
                    </p>
                    <p className="text-xs text-foreground/50 truncate">
                      {persona.demographic}
                    </p>
                  </div>
                </div>
                
                {/* Reveal Overlay */}
                <AnimatePresence>
                  {!isRevealed && (
                    <motion.div 
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-background/80 backdrop-blur-[1px] flex items-center justify-center z-10"
                    >
                      <span className="text-xs font-bold text-foreground/40 uppercase tracking-widest animate-pulse">Pending</span>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Outcome Badge */}
                {isRevealed && (
                  <div className={`absolute top-3 right-3
                    ${isApproved ? 'text-green-600' : 'text-red-600'}
                  `}>
                    {isApproved ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>
      
    </div>
  );
}
