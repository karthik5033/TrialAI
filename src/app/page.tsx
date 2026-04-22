"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Upload, Scale, RefreshCw, Briefcase, HeartPulse, Building, Scale as ScaleIcon, ChevronRight } from "lucide-react";
import Link from "next/link";

const CountUp = ({ value, decimals = 2 }: { value: number, decimals?: number }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime: number | null = null;
    const duration = 1500;
    
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 4);
      setCount(value * ease);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(value);
      }
    };
    
    requestAnimationFrame(animate);
  }, [value]);

  return <>{decimals === 0 ? Math.round(count).toLocaleString() : count.toFixed(decimals)}</>;
};

export default function LandingPage() {
  const [liveStats, setLiveStats] = useState({
    demographic_parity: 0,
    disparate_impact: 0,
    equal_opportunity: 0
  });

  useEffect(() => {
    fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name: 'COMPAS' })
    })
      .then(res => res.json())
      .then(data => {
        if (data.fairness_metrics) {
          setLiveStats(data.fairness_metrics);
        }
      })
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden selection:bg-gold/30">
      {/* Navigation */}
      <nav className="fixed top-0 w-full border-b border-border bg-background/80 backdrop-blur-md z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono font-bold text-lg tracking-tight">
            <ScaleIcon className="w-5 h-5 text-gold" />
            <span>TrialAI</span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/demo" className="text-sm font-medium text-foreground/80 hover:text-foreground transition-colors">
              Demo
            </Link>
            <Link 
              href="/trial/upload" 
              className="text-sm font-medium bg-foreground text-background px-4 py-2 rounded-md hover:bg-foreground/90 transition-colors"
            >
              Start Trial
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-[30%] -left-[10%] w-[50%] h-[50%] rounded-full bg-red-500/10 blur-[120px]" />
          <div className="absolute top-[20%] -right-[10%] w-[40%] h-[40%] rounded-full bg-blue-500/10 blur-[120px]" />
        </div>

        <div className="max-w-5xl mx-auto text-center relative z-10 pt-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="inline-block py-1 px-3 rounded-full bg-surface border border-border text-gold font-mono text-xs uppercase tracking-wider mb-6">
              v1.0 Production Ready
            </span>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tight mb-8 leading-[1.1]"
          >
            Put your AI on trial <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-foreground to-foreground/60">
              before the world does.
            </span>
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-lg md:text-xl text-foreground/60 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            An adversarial multi-agent courtroom simulation that audits AI models for hidden bias in hiring, lending, healthcare, and criminal justice.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link 
              href="/demo"
              className="group flex items-center gap-2 bg-gold hover:bg-gold/90 text-background font-medium px-8 py-4 rounded-lg transition-all"
            >
              Run a Demo Trial
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link 
              href="/trial/upload"
              className="flex items-center gap-2 bg-surface hover:bg-surface/80 border border-border text-foreground font-medium px-8 py-4 rounded-lg transition-all"
            >
              Upload Dataset
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Live Stats Section */}
      <section className="py-20 px-6 bg-surface/10 border-t border-border">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold mb-3 flex items-center justify-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
              Live Bias Intelligence — COMPAS Criminal Justice Dataset
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {/* Card 1: Demographic Parity */}
            <div className={`relative p-6 rounded-2xl border ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'bg-red-500/5 border-red-500/20' : 'bg-green-500/5 border-green-500/20'}`}>
              <div className="flex justify-end mb-2">
                <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                  {liveStats.demographic_parity === 0 ? '...' : (liveStats.demographic_parity < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-4xl font-black mb-2 ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'text-red-500' : 'text-green-500'}`}>
                {liveStats.demographic_parity > 0 ? <CountUp value={liveStats.demographic_parity} /> : "0.00"}
              </div>
              <div className="text-sm font-medium text-foreground/60">Demographic Parity Score</div>
            </div>

            {/* Card 2: Disparate Impact Ratio */}
            <div className={`relative p-6 rounded-2xl border ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'bg-red-500/5 border-red-500/20' : 'bg-green-500/5 border-green-500/20'}`}>
              <div className="flex justify-end mb-2">
                <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                  {liveStats.disparate_impact === 0 ? '...' : (liveStats.disparate_impact < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-4xl font-black mb-2 ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'text-red-500' : 'text-green-500'}`}>
                {liveStats.disparate_impact > 0 ? <CountUp value={liveStats.disparate_impact} /> : "0.00"}
              </div>
              <div className="text-sm font-medium text-foreground/60">Disparate Impact Ratio</div>
            </div>

            {/* Card 3: Equal Opportunity Score */}
            <div className={`relative p-6 rounded-2xl border ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'bg-red-500/5 border-red-500/20' : 'bg-green-500/5 border-green-500/20'}`}>
              <div className="flex justify-end mb-2">
                <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                  {liveStats.equal_opportunity === 0 ? '...' : (liveStats.equal_opportunity < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-4xl font-black mb-2 ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'text-red-500' : 'text-green-500'}`}>
                {liveStats.equal_opportunity > 0 ? <CountUp value={liveStats.equal_opportunity} /> : "0.00"}
              </div>
              <div className="text-sm font-medium text-foreground/60">Equal Opportunity Score</div>
            </div>

            {/* Card 4: Dataset Size */}
            <div className="relative p-6 rounded-2xl border bg-blue-500/5 border-blue-500/20">
              <div className="flex justify-end mb-2">
                <span className="text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider bg-blue-500/20 text-blue-500">
                  INFO
                </span>
              </div>
              <div className="text-4xl font-black mb-2 text-blue-500">
                <CountUp value={6907} decimals={0} />
              </div>
              <div className="text-sm font-medium text-foreground/60">defendants analyzed</div>
            </div>
          </div>

          <div className="text-center">
            <p className="text-red-600 font-bold text-lg md:text-xl">
              This model is currently deployed in 46 US states to make bail decisions.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 px-6 border-t border-border bg-surface/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">The Courtroom Protocol</h2>
            <p className="text-foreground/60 max-w-2xl mx-auto">A rigorous 3-step adversarial process to uncover, debate, and resolve model bias.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-[1px] bg-gradient-to-r from-transparent via-border to-transparent" />
            
            {[
              {
                icon: Upload,
                title: "1. Submit Evidence",
                desc: "Upload your dataset and model. We auto-detect sensitive demographic attributes.",
                color: "text-blue-500",
                bg: "bg-blue-500/10"
              },
              {
                icon: Scale,
                title: "2. The Trial",
                desc: "Prosecution and Defense LLM agents debate bias metrics while a synthetic jury experiences the model.",
                color: "text-red-500",
                bg: "bg-red-500/10"
              },
              {
                icon: RefreshCw,
                title: "3. Verdict & Reform",
                desc: "The Judge delivers a structured verdict and applies mitigation techniques for a fairer retrial.",
                color: "text-gold",
                bg: "bg-gold/10"
              }
            ].map((step, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="relative bg-background border border-border rounded-xl p-8 hover:border-border/80 transition-colors group"
              >
                <div className={`w-12 h-12 rounded-lg ${step.bg} ${step.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <step.icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                <p className="text-foreground/60 leading-relaxed">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-24 px-6 border-t border-border">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row gap-16 items-center">
            <div className="md:w-1/2">
              <h2 className="text-3xl md:text-4xl font-bold mb-6 leading-tight">
                High-stakes domains require high-scrutiny AI.
              </h2>
              <p className="text-foreground/60 text-lg mb-8 leading-relaxed">
                Static fairness dashboards are ignored. By putting models through an adversarial trial, we force explicit justification of disparate impact across critical sectors.
              </p>
              
              <div className="space-y-4">
                {[
                  { icon: Briefcase, text: "HR & Recruitment: Resume screening bias" },
                  { icon: Building, text: "Finance & Lending: Mortgage approval parity" },
                  { icon: HeartPulse, text: "Healthcare: Resource allocation fairness" },
                  { icon: ScaleIcon, text: "Criminal Justice: Recidivism prediction" },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 text-foreground/80">
                    <div className="w-8 h-8 rounded bg-surface border border-border flex items-center justify-center">
                      <item.icon className="w-4 h-4 text-gold" />
                    </div>
                    <span className="font-medium">{item.text}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="md:w-1/2 w-full">
              <div className="bg-surface border border-border rounded-xl p-6 font-mono text-sm relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 via-gold to-blue-500" />
                <div className="flex items-center justify-between mb-4 pb-4 border-b border-border/50">
                  <span className="text-foreground/50">Live Transcript Excerpt</span>
                  <span className="text-red-500 text-xs">● REC</span>
                </div>
                <div className="space-y-4">
                  <div>
                    <span className="text-red-500 font-bold">PROSECUTION: </span>
                    <span className="text-foreground/80">The data shows a disparate impact ratio of 0.62 for African American applicants. The model relies heavily on zip code proxies.</span>
                  </div>
                  <div>
                    <span className="text-blue-500 font-bold">DEFENSE: </span>
                    <span className="text-foreground/80">Zip code correlates with employment tenure in this dataset, which is a justified business necessity for loan approval.</span>
                  </div>
                  <div>
                    <span className="text-gold font-bold">JUDGE: </span>
                    <span className="text-foreground/80">Objection overruled. Proxy variables for protected classes violate demographic parity. Ordered mitigation: Feature Dropping and Retrial.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-gold/5" />
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h2 className="text-4xl font-bold mb-6">Ready to cross-examine your AI?</h2>
          <p className="text-xl text-foreground/60 mb-10">
            Run our pre-loaded COMPAS demo or upload your own dataset to see TrialAI in action.
          </p>
          <Link 
            href="/demo"
            className="inline-flex items-center gap-2 bg-foreground text-background font-medium px-8 py-4 rounded-lg hover:bg-foreground/90 transition-all"
          >
            Start the Trial
            <ChevronRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12 px-6 bg-surface/30 text-sm text-foreground/60">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2 font-mono font-bold text-foreground">
            <ScaleIcon className="w-4 h-4 text-gold" />
            <span>TrialAI</span>
          </div>
          
          <div className="flex gap-6">
            <Link href="#" className="hover:text-foreground transition-colors">About</Link>
            <Link href="/demo" className="hover:text-foreground transition-colors">Demo</Link>
            <Link href="/trial/upload" className="hover:text-foreground transition-colors">Upload</Link>
          </div>

          <div className="flex items-center gap-4">
            <a href="#" className="hover:text-foreground transition-colors font-medium">
              GitHub
            </a>
            <a href="#" className="hover:text-foreground transition-colors font-medium">
              LinkedIn
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
