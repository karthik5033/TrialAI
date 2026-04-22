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
    <div className="min-h-screen bg-background text-foreground pb-24">
      {/* Navigation */}
      <nav className="border-b border-border bg-surface sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-mono font-bold text-lg tracking-tight">
            <ScaleIcon className="w-5 h-5 text-gold" />
            <span>TrialAI</span>
          </Link>
          <div className="flex items-center gap-3 text-sm font-medium text-foreground/60">
            <span className={step >= 1 ? "text-foreground" : ""}>1. Upload Evidence</span>
            <ChevronRight className="w-4 h-4" />
            <span className={step >= 2 ? "text-foreground" : ""}>2. Configure</span>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 pt-12">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Submit Evidence for Trial</h1>
          <p className="text-foreground/60">Upload your dataset, trained model, and optionally the training script for a complete bias audit.</p>
        </motion.div>

        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-center gap-3 mb-6">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <p className="text-sm font-medium">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto"><X className="w-4 h-4" /></button>
          </motion.div>
        )}

        {/* Step 1: File Uploads */}
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div key="step1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">

              {/* CSV Upload */}
              <UploadZone
                label="Dataset"
                sublabel="CSV file with training/test data"
                icon={<FileText className="w-6 h-6" />}
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
                icon={<Cpu className="w-6 h-6" />}
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
                icon={<Code2 className="w-6 h-6" />}
                file={scriptFile}
                accept=".py"
                inputRef={scriptInputRef}
                onUpload={handleScriptUpload}
                onRemove={() => setScriptFile(null)}
                isDragging={isDragging === "script"}
                onDragStart={() => setIsDragging("script")}
                onDragEnd={() => setIsDragging(null)}
                color="amber"
              />

              {/* Next Button */}
              <div className="flex justify-end pt-4">
                <button
                  onClick={() => canProceedToConfig && setStep(2)}
                  disabled={!canProceedToConfig}
                  className={`flex items-center gap-2 px-8 py-3 rounded-lg font-medium transition-all
                    ${canProceedToConfig
                      ? 'bg-foreground text-background hover:bg-foreground/90 shadow-md'
                      : 'bg-surface border border-border text-foreground/40 cursor-not-allowed'}`}
                >
                  Configure Trial <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div key="step2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-8">

              {/* File Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <FileBadge icon={<FileText className="w-4 h-4" />} label="Dataset" name={csvFile!.name} color="blue" />
                <FileBadge icon={<Cpu className="w-4 h-4" />} label="Model" name={modelFile!.name} color="purple" />
                <FileBadge icon={<Code2 className="w-4 h-4" />} label="Script" name={scriptFile?.name || "Not provided"} color={scriptFile ? "amber" : "gray"} />
              </div>

              {/* Data Preview */}
              <div className="border border-border rounded-xl overflow-hidden bg-background">
                <div className="bg-surface px-4 py-3 border-b border-border flex items-center justify-between">
                  <h3 className="font-semibold text-sm">Data Preview (First 5 Rows)</h3>
                  <span className="text-xs text-foreground/50">{headers.length} columns</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs uppercase bg-surface/50 text-foreground/60 border-b border-border">
                      <tr>
                        {headers.map((h, i) => (
                          <th key={i} className="px-4 py-3 font-medium whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {previewRows.map((row, i) => (
                        <tr key={i} className="hover:bg-surface/30">
                          {row.map((cell, j) => (
                            <td key={j} className="px-4 py-2 whitespace-nowrap truncate max-w-[150px]">{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Configuration */}
              <div className="bg-surface border border-border rounded-xl p-6 space-y-8">
                {/* Sensitive Columns */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <h3 className="font-semibold">Sensitive Attributes</h3>
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">Auto-detected</span>
                  </div>
                  <p className="text-sm text-foreground/60 mb-4">
                    Select the columns that represent protected classes (e.g., race, gender, age). The Prosecution will probe these for bias.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {headers.map((h) => {
                      const isSensitive = sensitiveColumns.has(h);
                      return (
                        <button
                          key={h}
                          onClick={() => toggleSensitiveColumn(h)}
                          className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors flex items-center gap-1.5
                            ${isSensitive
                              ? 'bg-red-50 border-red-200 text-red-700'
                              : 'bg-background border-border text-foreground/70 hover:border-foreground/30'}`}
                        >
                          {isSensitive && <CheckCircle2 className="w-3.5 h-3.5" />}
                          {h}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="h-px bg-border w-full" />

                {/* Target Column */}
                <div>
                  <h3 className="font-semibold mb-2">Target Column</h3>
                  <p className="text-sm text-foreground/60 mb-3">What is your model predicting?</p>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full max-w-sm bg-background border border-border rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  >
                    <option value="" disabled>Select target column</option>
                    {headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-4">
                <button
                  onClick={() => setStep(1)}
                  className="flex items-center gap-2 px-6 py-3 rounded-lg font-medium text-foreground/60 hover:text-foreground transition-colors"
                >
                  ← Back
                </button>
                <button
                  onClick={handleBeginTrial}
                  disabled={!targetColumn || sensitiveColumns.size === 0 || isLoading}
                  className={`flex items-center gap-2 px-8 py-3 rounded-lg font-medium transition-all
                    ${targetColumn && sensitiveColumns.size > 0 && !isLoading
                      ? 'bg-foreground text-background hover:bg-foreground/90 shadow-md'
                      : 'bg-surface border border-border text-foreground/40 cursor-not-allowed'}`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing with Fairlearn + SHAP...
                    </>
                  ) : (
                    <>
                      Begin Trial
                      <ChevronRight className="w-5 h-5" />
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
  color: "blue" | "purple" | "amber" | "gray";
};

const colorMap = {
  blue: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-600", icon: "bg-blue-100 text-blue-600", dot: "bg-blue-500" },
  purple: { bg: "bg-purple-50", border: "border-purple-200", text: "text-purple-600", icon: "bg-purple-100 text-purple-600", dot: "bg-purple-500" },
  amber: { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-600", icon: "bg-amber-100 text-amber-600", dot: "bg-amber-500" },
  gray: { bg: "bg-gray-50", border: "border-gray-200", text: "text-gray-500", icon: "bg-gray-100 text-gray-500", dot: "bg-gray-400" },
};

function UploadZone({ label, sublabel, icon, file, accept, inputRef, onUpload, onRemove, isDragging, onDragStart, onDragEnd, required, color }: UploadZoneProps) {
  const c = colorMap[color];
  return (
    <div
      className={`border rounded-xl p-5 transition-all ${file ? `${c.bg} ${c.border}` : isDragging ? `border-${color}-400 bg-${color}-50/50 border-dashed` : 'border-border bg-surface hover:bg-surface/80 border-dashed'}`}
      onDragOver={(e) => { e.preventDefault(); onDragStart(); }}
      onDragLeave={(e) => { e.preventDefault(); onDragEnd(); }}
      onDrop={(e) => { e.preventDefault(); onDragEnd(); if (e.dataTransfer.files?.[0]) onUpload(e.dataTransfer.files[0]); }}
    >
      <input type="file" accept={accept} className="hidden" ref={inputRef as any} onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
      <div className="flex items-center gap-4">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${file ? c.icon : 'bg-gray-100 text-gray-400'}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold">{label}</h3>
            {required && <span className="text-xs text-red-500 font-medium">Required</span>}
            {!required && !file && <span className="text-xs text-foreground/40 font-medium">Optional</span>}
          </div>
          {file ? (
            <p className={`text-sm ${c.text} font-medium truncate`}>
              <CheckCircle2 className="w-3.5 h-3.5 inline mr-1" />
              {file.name} ({(file.size / 1024).toFixed(0)} KB)
            </p>
          ) : (
            <p className="text-sm text-foreground/50">{sublabel}</p>
          )}
        </div>
        {file ? (
          <button onClick={onRemove} className="p-2 text-foreground/50 hover:text-red-500 transition-colors rounded-md hover:bg-red-50">
            <X className="w-5 h-5" />
          </button>
        ) : (
          <button
            onClick={() => (inputRef as any).current?.click()}
            className="px-4 py-2 text-sm font-medium bg-background border border-border rounded-lg hover:bg-foreground hover:text-background transition-colors"
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
    <div className={`${c.bg} ${c.border} border rounded-lg p-3 flex items-center gap-3`}>
      <div className={`w-8 h-8 rounded flex items-center justify-center ${c.icon}`}>{icon}</div>
      <div className="min-w-0">
        <p className="text-xs text-foreground/50">{label}</p>
        <p className="text-sm font-medium truncate">{name}</p>
      </div>
    </div>
  );
}
