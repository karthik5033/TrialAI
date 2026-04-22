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
  ChevronLeft, 
  ChevronDown, 
  ChevronUp,
  ShieldAlert, 
  Loader2,
  PanelLeftClose, 
  PanelRightClose, 
  PanelBottomClose
} from "lucide-react";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import Link from "next/link";
import { useRouter } from "next/navigation";

// --- MOCK DATA FOR JURY/COUNTERFACTUALS (Since these aren't dynamic yet) ---

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
  "Your Honor, the prosecution calls the model to the stand. We submit Exhibit A: The model exhibits severe demographic disparity.",
  "Objection, Your Honor. The model does not explicitly use protected attributes as a feature. It relies on objective metrics to maximize accuracy.",
  "The defense's argument regarding accuracy is noted. However, the evidence shows a violation.",
  "Furthermore, counterfactual testing shows that flipping sensitive attributes decreases the risk score.",
  "We argue that altering this would drastically reduce the model's accuracy on the general population.",
  "Accuracy cannot come at the expense of protected classes.",
  "The Equal Opportunity metric shows a marginalized demographic has a much higher false positive rate.",
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

export default function NewTrialPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeTab, setActiveTab] = useState<"Features" | "Fairness" | "Counterfactuals" | "Code" | "Proxies">("Fairness");
  const [juryState, setJuryState] = useState<number>(0);
  const [currentChargeIndex, setCurrentChargeIndex] = useState(0);
  const [trialComplete, setTrialComplete] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [bottomCollapsed, setBottomCollapsed] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const hasStarted = useRef(false);

  // Analysis data from LocalStorage
  const [analysisLoading, setAnalysisLoading] = useState(true);
  const [fairnessMetrics, setFairnessMetrics] = useState({ demographicParity: 1.0, equalOpportunity: 1.0, disparateImpact: 1.0 });
  const [shapValues, setShapValues] = useState<any[]>([]);
  const [demographics, setDemographics] = useState<any[]>([]);
  
  const [datasetName, setDatasetName] = useState("Loading...");
  const [datasetRows, setDatasetRows] = useState(0);
  const [datasetFeatures, setDatasetFeatures] = useState(0);
  const [modelAccuracy, setModelAccuracy] = useState(0);
  const [modelType, setModelType] = useState("Unknown");
  const [sensitiveAttrs, setSensitiveAttrs] = useState<string[]>([]);
  const [juryPersonas, setJuryPersonas] = useState(FALLBACK_JURY_PERSONAS);
  const [juryLoading, setJuryLoading] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [codeAnalysis, setCodeAnalysis] = useState<any>(null);
  const [proxyFeatures, setProxyFeatures] = useState<any[]>([]);

  // Load analysis data from localStorage
  useEffect(() => {
    const rawAnalysis = localStorage.getItem("trialAnalysis");
    const rawDatasetName = localStorage.getItem("trialDatasetName");

    if (!rawAnalysis || !rawDatasetName) {
      router.push("/upload");
      return;
    }

    try {
      const data = JSON.parse(rawAnalysis);
      setDatasetName(rawDatasetName);
      setDatasetRows(data.rows || 0);
      setDatasetFeatures(data.features || 0);
      setModelAccuracy(data.model_accuracy || 0);
      setModelType(data.model_type || "Unknown");
      setSensitiveAttrs(data.sensitive_attributes || []);
      if (data.session_id) {
        setSessionId(data.session_id);
        localStorage.setItem("trialSessionId", data.session_id);
      }
      if (data.code_analysis) setCodeAnalysis(data.code_analysis);
      if (data.proxy_features) setProxyFeatures(data.proxy_features);

      if (data.fairness_metrics) {
        console.log("Parsed fairness_metrics:", data.fairness_metrics);
        setFairnessMetrics({
          demographicParity: Number(data.fairness_metrics.demographicParity ?? data.fairness_metrics.demographic_parity ?? 1.0),
          equalOpportunity: Number(data.fairness_metrics.equalOpportunity ?? data.fairness_metrics.equal_opportunity ?? 1.0),
          disparateImpact: Number(data.fairness_metrics.disparateImpact ?? data.fairness_metrics.disparate_impact ?? 1.0),
        });
      }
      if (data.shap_values) {
        setShapValues(
          data.shap_values
            .map((sv: any) => ({
              feature: sv.is_proxy ? `${sv.feature} (Proxy)` : sv.feature,
              importance: sv.importance,
              isProxy: sv.is_proxy
            }))
            .reverse()
        );
      }
      if (data.demographic_breakdown) {
        const entries = Object.entries(data.demographic_breakdown) as [string, number][];
        // Sort by value descending
        entries.sort((a, b) => b[1] - a[1]);
        
        const top3 = entries.slice(0, 3);
        const rest = entries.slice(3);
        
        const processedEntries = [...top3];
        if (rest.length > 0) {
          const otherSum = rest.reduce((sum, entry) => sum + entry[1], 0);
          processedEntries.push(["Other", otherSum]);
        }
        
        const CHART_COLORS = ["#EF4444", "#3B82F6", "#F59E0B", "#10B981"];
        
        setDemographics(
          processedEntries.map(([name, value], i) => ({
            name,
            value,
            color: CHART_COLORS[i % CHART_COLORS.length],
          }))
        );
      }
      setAnalysisLoading(false);

      // Generate dynamic jury personas (async, doesn't block the rest)
      (async () => {
        // Pre-write fallback so it's guaranteed to exist if the LLM fails
        localStorage.setItem("trialJury", JSON.stringify(FALLBACK_JURY_PERSONAS));
        try {
          const juryRes = await fetch("/api/generate-jury", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              datasetName: rawDatasetName,
              sensitiveAttributes: data.sensitive_attributes || [],
              targetColumn: data.target_column || "target",
              demographicBreakdown: data.demographic_breakdown || {},
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
      })();
    } catch (e) {
      console.error("Failed to parse analysis data", e);
      router.push("/upload");
    }
  }, [router]);

  // Bias Risk Score Logic
  const avgMetric = (fairnessMetrics.demographicParity + fairnessMetrics.equalOpportunity + fairnessMetrics.disparateImpact) / 3;
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

  // Memory Restore
  useEffect(() => {
    const savedState = localStorage.getItem("trialChatState");
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        if (parsed.messages && parsed.messages.length > 0) {
          setMessages(parsed.messages);
          setCurrentChargeIndex(parsed.currentChargeIndex || 0);
          setTrialComplete(true); // Ensure user is not stuck if they navigate away mid-trial
          setJuryState(parsed.juryState || 0);
          hasStarted.current = true;
        }
      } catch (e) {
        console.error("Failed to restore memory", e);
      }
    }
  }, []);

  // Memory Save
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("trialChatState", JSON.stringify({
        messages,
        currentChargeIndex,
        trialComplete,
        juryState
      }));
    }
  }, [messages, currentChargeIndex, trialComplete, juryState]);

  // Simulation Logic
  useEffect(() => {
    if (analysisLoading) return;
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
      const currentDatasetName = localStorage.getItem("trialDatasetName") || "Dataset";
      const currentAnalysis = JSON.parse(localStorage.getItem("trialAnalysis") || "{}");
      const currentSensitiveAttrs = currentAnalysis.sensitive_attributes || ["Race", "Gender", "Age"];

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
            body: JSON.stringify({ dataset: currentDatasetName, sensitiveAttributes: currentSensitiveAttrs, metric })
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
              dataset: currentDatasetName, 
              sensitiveAttributes: currentSensitiveAttrs, 
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
            body: JSON.stringify({ dataset: currentDatasetName, sensitiveAttributes: currentSensitiveAttrs, metric })
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
              dataset: currentDatasetName, 
              sensitiveAttributes: currentSensitiveAttrs, 
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
              dataset: currentDatasetName, 
              sensitiveAttributes: currentSensitiveAttrs, 
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
  }, [analysisLoading]);

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
        <div className={`${leftCollapsed ? 'w-[48px]' : 'w-[300px]'} border-r border-border bg-surface/50 flex flex-col overflow-hidden transition-all duration-300 shrink-0`}>
          <div className={`flex items-center ${leftCollapsed ? 'justify-center p-3' : 'justify-between p-6 pb-0 mb-6'}`}>
            {!leftCollapsed && (
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                <h2 className="font-bold text-lg">Case File</h2>
              </div>
            )}
            <button onClick={() => setLeftCollapsed(!leftCollapsed)} className="p-1 rounded-md hover:bg-background transition-colors" title={leftCollapsed ? 'Expand Case File' : 'Collapse Case File'}>
              {leftCollapsed ? <ChevronRight className="w-4 h-4 text-foreground/50" /> : <PanelLeftClose className="w-4 h-4 text-foreground/50" />}
            </button>
          </div>
          {leftCollapsed ? (
            <div className="flex flex-col items-center gap-4 mt-4">
              <FileText className="w-4 h-4 text-foreground/30" />
              <Scale className="w-4 h-4 text-foreground/30" />
            </div>
          ) : (
          <div className="px-6 pb-6 overflow-y-auto flex-1">

          <div className="space-y-6">
            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-1">Dataset</p>
              <p className="font-medium">{datasetName}</p>
              <div className="flex items-center gap-4 mt-2 text-sm text-foreground/70">
                <span>{datasetRows.toLocaleString()} Rows</span>
                <span>{datasetFeatures} Features</span>
              </div>
            </div>

            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-2">Sensitive Attributes</p>
              <div className="flex flex-wrap gap-2">
                {sensitiveAttrs.map(attr => (
                  <span key={attr} className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded border border-red-200 capitalize">
                    {attr}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs text-foreground/50 font-bold uppercase mb-1">Model Profile</p>
              <p className="font-medium text-sm">{modelType}</p>
              <p className="text-sm text-green-600 font-medium mt-1">Accuracy: {(modelAccuracy * 100).toFixed(1)}%</p>
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
                      <span className="capitalize">{d.name}</span>
                    </div>
                    <span className="font-mono text-foreground/60">{d.value.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          </div>
          )}
        </div>

        {/* CENTER PANEL: Courtroom */}
        <div className="flex-1 flex flex-col bg-background relative overflow-hidden">
          {/* Top Bar */}
          <div className="h-16 border-b border-border flex items-center px-6 justify-between bg-surface/80 backdrop-blur z-10 shrink-0">
            <div>
              <h1 className="font-bold text-lg tracking-tight flex items-center gap-2">
                Live Trial
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
                <Link href={`/trial/new/verdict`} className="bg-foreground text-background px-8 py-3 rounded-xl font-bold flex items-center gap-2 hover:bg-foreground/90 transition-colors shadow-lg">
                  <Gavel className="w-5 h-5" /> View Verdict
                </Link>
              </motion.div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL: Evidence Board */}
        <div className={`${rightCollapsed ? 'w-[48px]' : 'w-[380px]'} border-l border-border bg-surface/50 flex flex-col shrink-0 transition-all duration-300 overflow-hidden`}>
          <div className={`border-b border-border bg-background ${rightCollapsed ? 'p-3 flex justify-center' : 'p-4'}`}>
            {rightCollapsed ? (
              <button onClick={() => setRightCollapsed(false)} className="p-1 rounded-md hover:bg-surface transition-colors" title="Expand Evidence Board">
                <ChevronLeft className="w-4 h-4 text-foreground/50" />
              </button>
            ) : (
              <>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-bold">Evidence Board</h2>
                  <button onClick={() => setRightCollapsed(true)} className="p-1 rounded-md hover:bg-surface transition-colors" title="Collapse Evidence Board">
                    <PanelRightClose className="w-4 h-4 text-foreground/50" />
                  </button>
                </div>
                <div className="flex gap-2 bg-surface p-1 rounded-lg border border-border">
                  {(["Fairness", "Features", "Code", "Proxies", "Counterfactuals"] as const).map(tab => (
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
              </>
            )}
          </div>
          {!rightCollapsed && (
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
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.demographicParity < 0.8 ? 'text-red-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.demographicParity < 0.7 ? 'text-red-600' : fairnessMetrics.demographicParity < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.demographicParity}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.demographicParity < 0.8 ? 'Severe violation (Threshold: >0.80). Outcomes are disproportionate.' : 'Within acceptable threshold.'}</p>
                </div>
                
                <div className="p-4 bg-background border border-border rounded-xl">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold">Equal Opportunity</h3>
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.equalOpportunity < 0.8 ? 'text-amber-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.equalOpportunity < 0.7 ? 'text-red-600' : fairnessMetrics.equalOpportunity < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.equalOpportunity}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.equalOpportunity < 0.8 ? 'Moderate violation. True positive rates differ significantly across demographic groups.' : 'Within acceptable threshold.'}</p>
                </div>

                <div className="p-4 bg-background border border-border rounded-xl">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold">Disparate Impact</h3>
                    <AlertTriangle className={`w-4 h-4 ${fairnessMetrics.disparateImpact < 0.8 ? 'text-red-500' : 'text-green-500'}`} />
                  </div>
                  <div className={`text-3xl font-bold mb-1 ${fairnessMetrics.disparateImpact < 0.7 ? 'text-red-600' : fairnessMetrics.disparateImpact < 0.8 ? 'text-amber-600' : 'text-green-600'}`}>{fairnessMetrics.disparateImpact}</div>
                  <p className="text-xs text-foreground/60">{fairnessMetrics.disparateImpact < 0.8 ? 'Severe violation. Structural bias detected in the underlying dataset distributions.' : 'Within acceptable threshold.'}</p>
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
                          <Cell key={`cell-${index}`} fill={entry.isProxy ? "#EF4444" : "#3B82F6"} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>
            )}

            {activeTab === "Counterfactuals" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                {COUNTERFACTUALS.map(cf => (
                  <div key={cf.id} className="bg-background border border-border p-3 rounded-lg text-sm">
                    <p className="text-xs font-mono text-blue-600 bg-blue-50 inline-block px-1 rounded mb-2">{cf.attr}</p>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-foreground/60 line-through">{cf.original}</span>
                      <ArrowRight className="w-3 h-3 mx-2 text-foreground/40" />
                      <span className="font-semibold">{cf.flipped}</span>
                    </div>
                    <p className="text-xs text-amber-600 mt-2 bg-amber-50 px-2 py-1 rounded">{cf.change}</p>
                  </div>
                ))}
              </motion.div>
            )}

            {activeTab === "Code" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                {codeAnalysis ? (
                  <>
                    <div className={`p-4 rounded-xl border ${
                      codeAnalysis.risk_level === 'HIGH' ? 'bg-red-50 border-red-200' :
                      codeAnalysis.risk_level === 'MODERATE' ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                          codeAnalysis.risk_level === 'HIGH' ? 'bg-red-200 text-red-800' :
                          codeAnalysis.risk_level === 'MODERATE' ? 'bg-amber-200 text-amber-800' : 'bg-green-200 text-green-800'
                        }`}>{codeAnalysis.risk_level} RISK</span>
                        <span className="text-xs text-foreground/50">Model: {codeAnalysis.model_type_detected || modelType}</span>
                      </div>
                      <div className="flex gap-4 text-xs mt-2">
                        <span className={codeAnalysis.has_class_balancing ? 'text-green-600' : 'text-red-600'}>
                          {codeAnalysis.has_class_balancing ? '✓' : '✗'} Class Balancing
                        </span>
                        <span className={!codeAnalysis.uses_sensitive_features ? 'text-green-600' : 'text-amber-600'}>
                          {!codeAnalysis.uses_sensitive_features ? '✓' : '⚠'} Sensitive Features
                        </span>
                      </div>
                    </div>
                    {codeAnalysis.issues?.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-xs font-bold uppercase text-foreground/50">Issues Found</h4>
                        {codeAnalysis.issues.map((issue: string, i: number) => (
                          <div key={i} className="p-3 bg-background border border-border rounded-lg text-sm flex items-start gap-2">
                            <XCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                            <span>{issue}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {codeAnalysis.recommendations?.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-xs font-bold uppercase text-foreground/50">Recommendations</h4>
                        {codeAnalysis.recommendations.map((rec: string, i: number) => (
                          <div key={i} className="p-3 bg-background border border-border rounded-lg text-sm flex items-start gap-2">
                            <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                            <span>{rec}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-6 text-center text-foreground/50 text-sm">
                    <p className="mb-1">No training script was uploaded.</p>
                    <p className="text-xs">Upload a .py training script to enable LLM code analysis.</p>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === "Proxies" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <h3 className="text-sm font-semibold">Proxy Variable Detection</h3>
                <p className="text-xs text-foreground/60 mb-2">Features correlated (&gt;0.25) with sensitive attributes may act as proxies for protected classes.</p>
                {proxyFeatures.length > 0 ? (
                  proxyFeatures.map((pf, i) => (
                    <div key={i} className="p-3 bg-background border border-border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold">{pf.feature}</span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                          pf.correlation > 0.5 ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
                        }`}>{(pf.correlation * 100).toFixed(0)}% correlated</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div className={`h-1.5 rounded-full ${pf.correlation > 0.5 ? 'bg-red-500' : 'bg-amber-500'}`}
                          style={{ width: `${pf.correlation * 100}%` }} />
                      </div>
                      <p className="text-xs text-foreground/50 mt-1">Correlated with: {pf.corr_with}</p>
                    </div>
                  ))
                ) : (
                  <div className="p-6 text-center text-foreground/50 text-sm">
                    <p>No significant proxy variables detected.</p>
                    <p className="text-xs mt-1">Features with &gt;25% correlation to sensitive attributes would appear here.</p>
                  </div>
                )}
              </motion.div>
            )}
            </>
            )}
          </div>
          )}
        </div>
      </div>

      {/* BOTTOM PANEL: The Jury */}
      <div className={`${bottomCollapsed ? 'h-[44px]' : 'h-[180px]'} border-t border-border bg-surface shrink-0 overflow-hidden flex flex-col transition-all duration-300`}>
        <div className="flex items-center justify-between px-4 py-2.5 shrink-0">
          <h3 className="text-sm font-bold flex items-center gap-2">
            Synthetic Jury {!bottomCollapsed && <span className="text-xs font-normal text-foreground/50">Experiencing model decisions in real-time</span>}
          </h3>
          <button onClick={() => setBottomCollapsed(!bottomCollapsed)} className="p-1 rounded-md hover:bg-background transition-colors" title={bottomCollapsed ? 'Expand Jury' : 'Collapse Jury'}>
            {bottomCollapsed ? <ChevronUp className="w-4 h-4 text-foreground/50" /> : <PanelBottomClose className="w-4 h-4 text-foreground/50" />}
          </button>
        </div>
        <div className="flex gap-4 overflow-x-auto pb-4 px-4 hide-scrollbar">
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
