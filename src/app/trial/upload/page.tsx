"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, Code2, Bot, CheckCircle2, AlertCircle, X, ChevronRight, Scale as ScaleIcon, Loader2, Cpu } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function UploadPage() {
  const router = useRouter();
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [scriptFile, setScriptFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [step, setStep] = useState(1); // 1=upload, 2=configure

  const [headers, setHeaders] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<string[][]>([]);

  const [sensitiveColumns, setSensitiveColumns] = useState<Set<string>>(new Set());
  const [targetColumn, setTargetColumn] = useState<string>("");

  const csvInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);
  const scriptInputRef = useRef<HTMLInputElement>(null);

  const SENSITIVE_KEYWORDS = ["gender", "sex", "race", "age", "ethnicity", "nationality", "religion", "zipcode", "income", "marital"];

  const parseCSV = (text: string) => {
    const lines = text.split(/\r?\n/).filter(line => line.trim() !== "");
    if (lines.length === 0) return;
    const splitRegex = /,(?=(?:(?:[^"]*"){2})*[^"]*$)/;
    const parsedHeaders = lines[0].split(splitRegex).map(h => h.trim().replace(/^"|"$/g, ""));
    const parsedRows = lines.slice(1, 6).map(line =>
      line.split(splitRegex).map(cell => cell.trim().replace(/^"|"$/g, ""))
    );
    setHeaders(parsedHeaders);
    setPreviewRows(parsedRows);
    const detected = new Set<string>();
    parsedHeaders.forEach(header => {
      const lowerHeader = header.toLowerCase();
      if (SENSITIVE_KEYWORDS.some(kw => lowerHeader.includes(kw))) {
        detected.add(header);
      }
    });
    setSensitiveColumns(detected);
  };

  const handleCSVUpload = (uploadedFile: File) => {
    setError(null);
    if (!uploadedFile.name.endsWith('.csv')) { setError("Please upload a valid CSV file."); return; }
    if (uploadedFile.size > 50 * 1024 * 1024) { setError("File size exceeds 50MB limit."); return; }
    setCsvFile(uploadedFile);
    const reader = new FileReader();
    reader.onload = (e) => parseCSV(e.target?.result as string);
    reader.readAsText(uploadedFile);
  };

  const handleModelUpload = (uploadedFile: File) => {
    setError(null);
    const ext = uploadedFile.name.split('.').pop()?.toLowerCase();
    if (!['pkl', 'joblib', 'pickle'].includes(ext || '')) {
      setError("Please upload a .pkl or .joblib model file."); return;
    }
    setModelFile(uploadedFile);
  };

  const handleScriptUpload = (uploadedFile: File) => {
    setError(null);
    if (!uploadedFile.name.endsWith('.py')) { setError("Please upload a .py Python file."); return; }
    setScriptFile(uploadedFile);
  };

  const toggleSensitiveColumn = (col: string) => {
    const newSet = new Set(sensitiveColumns);
    if (newSet.has(col)) newSet.delete(col); else newSet.add(col);
    setSensitiveColumns(newSet);
  };

  const handleBeginTrial = async () => {
    if (!csvFile || !modelFile || !targetColumn || sensitiveColumns.size === 0) return;
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", csvFile);
      formData.append("model_file", modelFile);
      formData.append("target_column", targetColumn);
      formData.append("sensitive_attributes", Array.from(sensitiveColumns).join(","));
      if (scriptFile) formData.append("training_script", scriptFile);

      const res = await fetch("/api/full-analysis", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Analysis failed.");
      }

      localStorage.setItem("trialAnalysis", JSON.stringify(data));
      localStorage.setItem("trialDatasetName", csvFile.name.replace(/\.csv$/i, ""));
      if (data.session_id) localStorage.setItem("trialSessionId", data.session_id);
      localStorage.removeItem("trialChatState");

      router.push("/trial/new");
    } catch (err: any) {
      console.error("Trial error:", err);
      setError(err.message || "Analysis failed. Please check your files and try again.");
      setIsLoading(false);
    }
  };

  const allFilesReady = csvFile && modelFile;
  const canProceedToConfig = allFilesReady && headers.length > 0;

  return (
    <div className="relative min-h-screen text-white overflow-hidden selection:bg-gold/30 pb-32 font-sans">
      {/* Video Background */}
      <div className="fixed inset-0 w-full h-full z-[-2]">
        <video 
          autoPlay 
          loop 
          muted 
          playsInline 
          className="w-full h-full object-cover scale-105"
        >
          <source src="/landing-video.mp4" type="video/mp4" />
        </video>
      </div>

      {/* Cinematic Overlay */}
      <div className="fixed inset-0 bg-black/40 z-[-1]" />
      <div className="fixed inset-0 bg-gradient-to-b from-black/10 via-black/50 to-black/90 z-[-1]" />

      {/* Step Indicator (Glassmorphism) */}
      <div className="fixed top-16 left-0 w-full border-b border-white/10 bg-black/20 backdrop-blur-xl z-40">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-end gap-5 text-sm font-bold text-white/50 uppercase tracking-widest">
          <span className={`transition-all ${step >= 1 ? "text-gold drop-shadow-[0_0_10px_rgba(212,175,55,0.8)]" : ""}`}>1. Upload Evidence</span>
          <ChevronRight className="w-5 h-5 text-white/20" />
          <span className={`transition-all ${step >= 2 ? "text-gold drop-shadow-[0_0_10px_rgba(212,175,55,0.8)]" : ""}`}>2. Configure</span>
        </div>
      </div>

      <main className="relative z-10 max-w-4xl mx-auto px-6 pt-40">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12 text-center">
          <h1 className="text-4xl md:text-6xl font-black mb-6 tracking-tight drop-shadow-lg">Submit Evidence for Trial</h1>
          <p className="text-white/70 text-xl font-light max-w-2xl mx-auto">Upload your dataset, trained model, and optionally the training script for a complete adversarial bias audit.</p>
        </motion.div>

        {error && (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
            className="bg-red-500/20 backdrop-blur-xl border border-red-500/50 text-red-100 p-5 rounded-2xl flex items-center gap-4 mb-10 shadow-[0_0_40px_rgba(239,68,68,0.2)]">
            <AlertCircle className="w-6 h-6 shrink-0 text-red-400" />
            <p className="text-base font-medium">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto hover:text-white transition-colors"><X className="w-5 h-5" /></button>
          </motion.div>
        )}

        {/* Step 1: File Uploads */}
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div key="step1" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20, filter: "blur(10px)" }} className="space-y-6">

              {/* CSV Upload */}
              <UploadZone
                label="Dataset"
                sublabel="CSV file with training/test data"
                icon={<FileText className="w-7 h-7" />}
                file={csvFile}
                accept=".csv"
                inputRef={csvInputRef}
                onUpload={handleCSVUpload}
                onRemove={() => { setCsvFile(null); setHeaders([]); setPreviewRows([]); }}
                isDragging={isDragging === "csv"}
                onDragStart={() => setIsDragging("csv")}
                onDragEnd={() => setIsDragging(null)}
                required
                color="blue"
              />

              {/* Model Upload */}
              <UploadZone
                label="Trained Model"
                sublabel=".pkl or .joblib (sklearn-compatible)"
                icon={<Cpu className="w-7 h-7" />}
                file={modelFile}
                accept=".pkl,.joblib,.pickle"
                inputRef={modelInputRef}
                onUpload={handleModelUpload}
                onRemove={() => setModelFile(null)}
                isDragging={isDragging === "model"}
                onDragStart={() => setIsDragging("model")}
                onDragEnd={() => setIsDragging(null)}
                required
                color="purple"
              />

              {/* Script Upload (Optional) */}
              <UploadZone
                label="Training Script"
                sublabel=".py file (optional — enables code analysis & auto-retrain)"
                icon={<Code2 className="w-7 h-7" />}
                file={scriptFile}
                accept=".py"
                inputRef={scriptInputRef}
                onUpload={handleScriptUpload}
                onRemove={() => setScriptFile(null)}
                isDragging={isDragging === "script"}
                onDragStart={() => setIsDragging("script")}
                onDragEnd={() => setIsDragging(null)}
                color="gold"
              />

              {/* Next Button */}
              <div className="flex justify-end pt-8">
                <button
                  onClick={() => canProceedToConfig && setStep(2)}
                  disabled={!canProceedToConfig}
                  className={`flex items-center gap-3 px-12 py-5 rounded-2xl font-black text-lg transition-all
                    ${canProceedToConfig
                      ? 'bg-gold hover:bg-yellow-500 text-black shadow-[0_0_40px_rgba(212,175,55,0.4)] hover:shadow-[0_0_60px_rgba(212,175,55,0.6)] hover:-translate-y-1'
                      : 'bg-white/5 border border-white/10 text-white/30 cursor-not-allowed'}`}
                >
                  Configure Trial <ChevronRight className="w-6 h-6" />
                </button>
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div key="step2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20, filter: "blur(10px)" }} className="space-y-10">

              {/* File Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <FileBadge icon={<FileText className="w-5 h-5" />} label="Dataset" name={csvFile!.name} color="blue" />
                <FileBadge icon={<Cpu className="w-5 h-5" />} label="Model" name={modelFile!.name} color="purple" />
                <FileBadge icon={<Code2 className="w-5 h-5" />} label="Script" name={scriptFile?.name || "Not provided"} color={scriptFile ? "gold" : "gray"} />
              </div>

              {/* Data Preview */}
              <div className="bg-black/40 backdrop-blur-2xl border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
                <div className="bg-white/5 px-8 py-5 border-b border-white/10 flex items-center justify-between">
                  <h3 className="font-bold text-lg text-white tracking-wide">Data Preview (First 5 Rows)</h3>
                  <span className="text-xs font-mono font-bold text-white/60 bg-white/10 px-4 py-1.5 rounded-full border border-white/5">{headers.length} COLUMNS</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs uppercase bg-black/40 text-white/50 border-b border-white/10">
                      <tr>
                        {headers.map((h, i) => (
                          <th key={i} className="px-8 py-5 font-black whitespace-nowrap tracking-wider">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {previewRows.map((row, i) => (
                        <tr key={i} className="hover:bg-white/5 transition-colors">
                          {row.map((cell, j) => (
                            <td key={j} className="px-8 py-4 whitespace-nowrap truncate max-w-[200px] text-white/80 font-light">{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Configuration */}
              <div className="bg-black/40 backdrop-blur-2xl border border-white/10 rounded-3xl p-10 space-y-10 shadow-2xl">
                {/* Sensitive Columns */}
                <div>
                  <div className="flex items-center gap-4 mb-5">
                    <h3 className="text-2xl font-black text-white">Sensitive Attributes</h3>
                    <span className="text-xs bg-gold/20 text-gold border border-gold/30 px-3 py-1 rounded-full font-bold uppercase tracking-widest drop-shadow-[0_0_10px_rgba(212,175,55,0.5)]">Auto-detected</span>
                  </div>
                  <p className="text-lg text-white/60 mb-8 font-light">
                    Select the columns that represent protected classes (e.g., race, gender, age). The Prosecution will probe these for bias.
                  </p>
                  <div className="flex flex-wrap gap-4">
                    {headers.map((h) => {
                      const isSensitive = sensitiveColumns.has(h);
                      return (
                        <button
                          key={h}
                          onClick={() => toggleSensitiveColumn(h)}
                          className={`px-5 py-3 rounded-xl text-base font-bold border transition-all flex items-center gap-3
                            ${isSensitive
                              ? 'bg-red-500/20 border-red-500/50 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.2)]'
                              : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10 hover:border-white/30 hover:text-white'}`}
                        >
                          {isSensitive && <CheckCircle2 className="w-5 h-5" />}
                          {h}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent w-full" />

                {/* Target Column */}
                <div>
                  <h3 className="text-2xl font-black text-white mb-3">Target Column</h3>
                  <p className="text-lg text-white/60 mb-6 font-light">What is your model predicting?</p>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full max-w-xl bg-black/50 border border-white/20 rounded-2xl px-6 py-4 text-lg text-white focus:outline-none focus:ring-2 focus:ring-gold/50 focus:border-gold transition-all appearance-none cursor-pointer hover:bg-white/5"
                    style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='white'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 1.5rem center', backgroundSize: '1.5em' }}
                  >
                    <option value="" disabled className="bg-black text-white/50">Select target column</option>
                    {headers.map(h => <option key={h} value={h} className="bg-black text-white">{h}</option>)}
                  </select>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-8">
                <button
                  onClick={() => setStep(1)}
                  className="flex items-center gap-3 px-8 py-5 rounded-2xl font-bold text-lg text-white/50 hover:text-white hover:bg-white/5 transition-all"
                >
                  ← Back to Uploads
                </button>
                <button
                  onClick={handleBeginTrial}
                  disabled={!targetColumn || sensitiveColumns.size === 0 || isLoading}
                  className={`flex items-center gap-3 px-12 py-5 rounded-2xl font-black text-lg transition-all
                    ${targetColumn && sensitiveColumns.size > 0 && !isLoading
                      ? 'bg-gold hover:bg-yellow-500 text-black shadow-[0_0_40px_rgba(212,175,55,0.4)] hover:shadow-[0_0_60px_rgba(212,175,55,0.6)] hover:-translate-y-1'
                      : 'bg-white/5 border border-white/10 text-white/30 cursor-not-allowed'}`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Summoning Courtroom...
                    </>
                  ) : (
                    <>
                      Begin Trial
                      <ChevronRight className="w-6 h-6" />
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

// ─── Reusable Upload Zone ─────────────────────────────────────────────────

type UploadZoneProps = {
  label: string;
  sublabel: string;
  icon: React.ReactNode;
  file: File | null;
  accept: string;
  inputRef: React.RefObject<HTMLInputElement>;
  onUpload: (f: File) => void;
  onRemove: () => void;
  isDragging: boolean;
  onDragStart: () => void;
  onDragEnd: () => void;
  required?: boolean;
  color: "blue" | "purple" | "gold" | "gray";
};

const colorMap = {
  blue: { bg: "bg-blue-500/10", border: "border-blue-500/30", text: "text-blue-400", icon: "bg-blue-500/20 text-blue-400", shadow: "shadow-[0_0_20px_rgba(59,130,246,0.15)]", dragBorder: "border-blue-500", dragBg: "bg-blue-500/10" },
  purple: { bg: "bg-purple-500/10", border: "border-purple-500/30", text: "text-purple-400", icon: "bg-purple-500/20 text-purple-400", shadow: "shadow-[0_0_20px_rgba(168,85,247,0.15)]", dragBorder: "border-purple-500", dragBg: "bg-purple-500/10" },
  gold: { bg: "bg-gold/10", border: "border-gold/30", text: "text-gold", icon: "bg-gold/20 text-gold", shadow: "shadow-[0_0_20px_rgba(212,175,55,0.15)]", dragBorder: "border-gold", dragBg: "bg-gold/10" },
  gray: { bg: "bg-white/5", border: "border-white/10", text: "text-white/50", icon: "bg-white/10 text-white/50", shadow: "", dragBorder: "border-white/30", dragBg: "bg-white/10" },
};

function UploadZone({ label, sublabel, icon, file, accept, inputRef, onUpload, onRemove, isDragging, onDragStart, onDragEnd, required, color }: UploadZoneProps) {
  const c = colorMap[color];
  return (
    <div
      className={`border rounded-3xl p-8 transition-all duration-300 backdrop-blur-2xl ${file ? `${c.bg} ${c.border} ${c.shadow}` : isDragging ? `${c.dragBorder} ${c.dragBg} border-dashed scale-[1.02] shadow-2xl` : 'border-white/10 bg-black/40 hover:bg-black/60 hover:border-white/20 border-dashed'}`}
      onDragOver={(e) => { e.preventDefault(); onDragStart(); }}
      onDragLeave={(e) => { e.preventDefault(); onDragEnd(); }}
      onDrop={(e) => { e.preventDefault(); onDragEnd(); if (e.dataTransfer.files?.[0]) onUpload(e.dataTransfer.files[0]); }}
    >
      <input type="file" accept={accept} className="hidden" ref={inputRef as any} onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
      <div className="flex items-center gap-8">
        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center border border-white/5 ${file ? c.icon : 'bg-white/5 text-white/40'}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="font-black text-xl text-white">{label}</h3>
            {required && <span className="text-[10px] bg-red-500/20 text-red-400 border border-red-500/30 px-3 py-1 rounded-full font-bold uppercase tracking-widest">Required</span>}
            {!required && !file && <span className="text-[10px] bg-white/10 text-white/40 border border-white/10 px-3 py-1 rounded-full font-bold uppercase tracking-widest">Optional</span>}
          </div>
          {file ? (
            <p className={`text-base ${c.text} font-bold truncate flex items-center gap-2`}>
              <CheckCircle2 className="w-5 h-5" />
              {file.name} <span className="text-white/40 font-normal ml-2">({(file.size / 1024).toFixed(0)} KB)</span>
            </p>
          ) : (
            <p className="text-base text-white/50 font-light">{sublabel}</p>
          )}
        </div>
        {file ? (
          <button onClick={onRemove} className="p-4 text-white/40 hover:text-red-400 transition-all rounded-xl hover:bg-red-500/10 hover:scale-110">
            <X className="w-6 h-6" />
          </button>
        ) : (
          <button
            onClick={() => (inputRef as any).current?.click()}
            className="px-8 py-4 text-base font-bold bg-white text-black rounded-2xl hover:bg-gray-200 transition-all shadow-[0_0_20px_rgba(255,255,255,0.2)] hover:shadow-[0_0_30px_rgba(255,255,255,0.4)] hover:-translate-y-1"
          >
            Browse
          </button>
        )}
      </div>
    </div>
  );
}

function FileBadge({ icon, label, name, color }: { icon: React.ReactNode; label: string; name: string; color: string }) {
  const c = colorMap[color as keyof typeof colorMap] || colorMap.gray;
  return (
    <div className={`${c.bg} ${c.border} border rounded-2xl p-5 flex items-center gap-4 backdrop-blur-2xl ${c.shadow}`}>
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center border border-white/5 ${c.icon}`}>{icon}</div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-widest text-white/50 font-bold mb-1">{label}</p>
        <p className="text-base font-bold text-white truncate">{name}</p>
      </div>
    </div>
  );
}
