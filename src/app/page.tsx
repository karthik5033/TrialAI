"use client";

import { useEffect, useState } from "react";
import { ArrowRight, Upload, Scale, RefreshCw, Briefcase, HeartPulse, Building, Scale as ScaleIcon, ChevronRight, Activity, Code2, Fingerprint } from "lucide-react";
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
    setLiveStats({
      demographic_parity: 0.62,
      disparate_impact: 0.58,
      equal_opportunity: 0.75
    });
  }, []);

  return (
    <div className="relative min-h-screen text-white selection:bg-gold/30 font-sans">
      {/* Video Background */}
      <div className="fixed inset-0 w-full h-full z-[-2] overflow-hidden">
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
      <div className="fixed inset-0 bg-black/30 z-[-1]" />
      <div className="fixed inset-0 bg-gradient-to-b from-black/10 via-black/30 to-black/80 z-[-1]" />

      {/* Hero Section */}
      <section id="about" className="relative pt-32 pb-20 px-6">
        <div className="max-w-5xl mx-auto text-center relative z-10 pt-20">
          <div>
            <span className="inline-flex items-center gap-2 py-1.5 px-4 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-gold font-mono text-xs uppercase tracking-widest mb-8 shadow-2xl">
              <span className="w-2 h-2 rounded-full bg-gold animate-pulse" />
              v1.0 Production Ready
            </span>
          </div>
          
          <h1 className="text-6xl md:text-8xl font-black tracking-tight mb-8 leading-[1.05]">
            Put your AI on trial. <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-white/50">
              Before the world does.
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-white/70 max-w-3xl mx-auto mb-12 leading-relaxed font-light">
            An adversarial multi-agent courtroom simulation that interrogates AI models for hidden bias in hiring, lending, healthcare, and criminal justice.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-5">
            <Link 
              href="/demo"
              className="group flex items-center gap-3 bg-gold hover:bg-yellow-500 text-black font-bold px-10 py-5 rounded-xl transition-all shadow-[0_0_40px_rgba(212,175,55,0.3)] hover:shadow-[0_0_60px_rgba(212,175,55,0.5)] hover:-translate-y-1"
            >
              Run a Demo Trial
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link 
              href="/trial/upload"
              className="flex items-center gap-3 bg-white/5 hover:bg-white/10 backdrop-blur-md border border-white/20 text-white font-bold px-10 py-5 rounded-xl transition-all hover:-translate-y-1"
            >
              Upload Dataset
            </Link>
          </div>
        </div>
      </section>

      {/* Live Stats Section */}
      <section className="py-24 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-2xl md:text-3xl font-bold mb-3 flex items-center justify-center gap-3 text-white/90">
              <Activity className="w-6 h-6 text-red-500 animate-pulse" />
              Live Bias Intelligence — COMPAS Dataset
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {/* Card 1 */}
            <div className={`relative p-8 rounded-2xl backdrop-blur-xl border ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'bg-red-500/10 border-red-500/30' : 'bg-green-500/10 border-green-500/30'}`}>
              <div className="flex justify-end mb-4">
                <span className={`text-[10px] font-black px-3 py-1.5 rounded-full uppercase tracking-widest ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                  {liveStats.demographic_parity === 0 ? '...' : (liveStats.demographic_parity < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-5xl font-black mb-3 ${liveStats.demographic_parity > 0 && liveStats.demographic_parity < 0.8 ? 'text-red-400' : 'text-green-400'}`}>
                <CountUp value={liveStats.demographic_parity} />
              </div>
              <div className="text-sm font-medium text-white/60">Demographic Parity</div>
            </div>

            {/* Card 2 */}
            <div className={`relative p-8 rounded-2xl backdrop-blur-xl border ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'bg-red-500/10 border-red-500/30' : 'bg-green-500/10 border-green-500/30'}`}>
              <div className="flex justify-end mb-4">
                <span className={`text-[10px] font-black px-3 py-1.5 rounded-full uppercase tracking-widest ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                  {liveStats.disparate_impact === 0 ? '...' : (liveStats.disparate_impact < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-5xl font-black mb-3 ${liveStats.disparate_impact > 0 && liveStats.disparate_impact < 0.8 ? 'text-red-400' : 'text-green-400'}`}>
                <CountUp value={liveStats.disparate_impact} />
              </div>
              <div className="text-sm font-medium text-white/60">Disparate Impact</div>
            </div>

            {/* Card 3 */}
            <div className={`relative p-8 rounded-2xl backdrop-blur-xl border ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'bg-red-500/10 border-red-500/30' : 'bg-green-500/10 border-green-500/30'}`}>
              <div className="flex justify-end mb-4">
                <span className={`text-[10px] font-black px-3 py-1.5 rounded-full uppercase tracking-widest ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                  {liveStats.equal_opportunity === 0 ? '...' : (liveStats.equal_opportunity < 0.8 ? 'FAIL' : 'PASS')}
                </span>
              </div>
              <div className={`text-5xl font-black mb-3 ${liveStats.equal_opportunity > 0 && liveStats.equal_opportunity < 0.8 ? 'text-red-400' : 'text-green-400'}`}>
                <CountUp value={liveStats.equal_opportunity} />
              </div>
              <div className="text-sm font-medium text-white/60">Equal Opportunity</div>
            </div>

            {/* Card 4 */}
            <div className="relative p-8 rounded-2xl backdrop-blur-xl border bg-blue-500/10 border-blue-500/30">
              <div className="flex justify-end mb-4">
                <span className="text-[10px] font-black px-3 py-1.5 rounded-full uppercase tracking-widest bg-blue-500/20 text-blue-400">
                  INFO
                </span>
              </div>
              <div className="text-5xl font-black mb-3 text-blue-400">
                <CountUp value={6907} decimals={0} />
              </div>
              <div className="text-sm font-medium text-white/60">Defendants Analyzed</div>
            </div>
          </div>
        </div>
      </section>

      {/* Protocol Section */}
      <section className="py-32 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl font-black mb-6">The Courtroom Protocol</h2>
            <p className="text-white/60 max-w-2xl mx-auto text-xl font-light">A rigorous 3-step adversarial process to uncover, debate, and resolve model bias.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {[
              { icon: Upload, title: "1. Submit Evidence", desc: "Upload your dataset and model. We auto-detect sensitive demographic attributes.", color: "text-blue-400", bg: "bg-blue-500/20 border-blue-500/30" },
              { icon: Scale, title: "2. The Trial", desc: "Prosecution and Defense LLM agents debate bias metrics while a synthetic jury experiences the model.", color: "text-red-400", bg: "bg-red-500/20 border-red-500/30" },
              { icon: RefreshCw, title: "3. Verdict & Reform", desc: "The Judge delivers a structured verdict and applies mitigation techniques for a fairer retrial.", color: "text-gold", bg: "bg-gold/20 border-gold/30" }
            ].map((step, i) => (
              <div 
                key={i}
                className="relative bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-10 hover:bg-black/60 hover:border-white/20 transition-all group"
              >
                <div className={`w-16 h-16 rounded-xl border ${step.bg} ${step.color} flex items-center justify-center mb-8 group-hover:scale-110 transition-transform`}>
                  <step.icon className="w-8 h-8" />
                </div>
                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                <p className="text-white/60 leading-relaxed font-light text-lg">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-32 px-6 relative z-10 border-t border-white/10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row gap-20 items-center">
          <div className="md:w-1/2">
            <h2 className="text-4xl md:text-5xl font-black mb-8 leading-tight">High-stakes domains require high-scrutiny AI.</h2>
            <div className="space-y-6">
              {[
                { icon: Briefcase, text: "HR & Recruitment: Resume screening bias" },
                { icon: Building, text: "Finance & Lending: Mortgage approval parity" },
                { icon: HeartPulse, text: "Healthcare: Resource allocation fairness" },
                { icon: ScaleIcon, text: "Criminal Justice: Recidivism prediction" },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-4 text-white/80 text-lg">
                  <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <item.icon className="w-6 h-6 text-gold" />
                  </div>
                  <span className="font-medium">{item.text}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="md:w-1/2 w-full">
            <div className="bg-black/60 backdrop-blur-2xl border border-white/10 rounded-2xl p-8 font-mono text-sm relative overflow-hidden shadow-2xl">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 via-gold to-blue-500" />
              <div className="flex items-center justify-between mb-6 pb-6 border-b border-white/10">
                <span className="text-white/40 uppercase tracking-widest text-xs font-bold">Live Transcript Excerpt</span>
                <span className="text-red-500 text-xs flex items-center gap-2 font-bold animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-red-500" /> REC
                </span>
              </div>
              <div className="space-y-6 text-base leading-relaxed">
                <div><span className="text-red-400 font-bold">PROSECUTION: </span><span className="text-white/70">The data shows a disparate impact ratio of 0.62 for African American applicants.</span></div>
                <div><span className="text-blue-400 font-bold">DEFENSE: </span><span className="text-white/70">Zip code correlates with employment tenure in this dataset.</span></div>
                <div><span className="text-gold font-bold">JUDGE: </span><span className="text-white/70">Objection overruled. Proxy variables for protected classes violate demographic parity.</span></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features & Documentation Section */}
      <section className="py-32 px-6 relative z-10 border-t border-white/10 bg-black/40">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl font-black mb-6">Platform Capabilities</h2>
            <p className="text-white/60 max-w-2xl mx-auto text-xl font-light">Comprehensive tools for auditing, mitigating, and documenting AI fairness.</p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-24">
            {[
              { title: "Adversarial LLMs", desc: "Multi-agent debate system that interrogates model decisions and uncovers proxy biases.", icon: Activity, color: "text-red-400" },
              { title: "Auto-Remediation", desc: "Automated script patching using reweighing, threshold adjustment, and fairness constraints.", icon: Code2, color: "text-blue-400" },
              { title: "SHAP Fingerprinting", desc: "Deep feature importance analysis to visualize exactly which attributes drive biased outcomes.", icon: Fingerprint, color: "text-gold" },
              { title: "Metric Tradeoffs", desc: "Interactive visualization of the accuracy vs. fairness tradeoff to find the optimal balance.", icon: ScaleIcon, color: "text-green-400" }
            ].map((feature, i) => (
              <div key={i} className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-6 hover:-translate-y-1 transition-all">
                <div className={`w-12 h-12 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center mb-6 ${feature.color}`}>
                  <feature.icon className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                <p className="text-sm text-white/50 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>

          {/* Documentation Section */}
          <div className="bg-gradient-to-r from-blue-900/20 via-black/40 to-red-900/20 rounded-3xl border border-white/10 p-10 md:p-16 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-gold/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
            
            <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-10">
              <div className="md:w-1/2">
                <h2 className="text-3xl font-black mb-4">Technical Documentation</h2>
                <p className="text-white/60 text-lg mb-8 leading-relaxed">
                  Explore our comprehensive guides, API references, and architectural deep-dives to integrate TrialAI into your ML pipeline.
                </p>
                <div className="flex flex-col gap-4">
                  <a href="#" className="group flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors">
                    <span className="font-medium text-white/80 group-hover:text-white transition-colors">Quick Start Guide</span>
                    <ChevronRight className="w-5 h-5 text-white/40 group-hover:text-white group-hover:translate-x-1 transition-all" />
                  </a>
                  <a href="#" className="group flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors">
                    <span className="font-medium text-white/80 group-hover:text-white transition-colors">Supported Fairness Metrics</span>
                    <ChevronRight className="w-5 h-5 text-white/40 group-hover:text-white group-hover:translate-x-1 transition-all" />
                  </a>
                  <a href="#" className="group flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors">
                    <span className="font-medium text-white/80 group-hover:text-white transition-colors">Auto-Remediation API</span>
                    <ChevronRight className="w-5 h-5 text-white/40 group-hover:text-white group-hover:translate-x-1 transition-all" />
                  </a>
                </div>
              </div>
              <div className="md:w-1/2 flex justify-center">
                <div className="w-full max-w-sm aspect-square relative">
                  <div className="absolute inset-0 border-2 border-dashed border-white/20 rounded-full animate-[spin_60s_linear_infinite]" />
                  <div className="absolute inset-4 border border-white/10 rounded-full animate-[spin_40s_linear_infinite_reverse]" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Code2 className="w-16 h-16 text-gold/50" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-12 px-6 bg-black/80 backdrop-blur-xl text-sm text-white/50 relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3 font-mono font-bold text-white text-lg">
            <ScaleIcon className="w-5 h-5 text-gold" />
            <span>TrialAI</span>
          </div>
          <div className="flex gap-8">
            <Link href="#about" className="hover:text-white transition-colors">About</Link>
            <Link href="/demo" className="hover:text-white transition-colors">Demo</Link>
            <Link href="/trial/upload" className="hover:text-white transition-colors">Upload</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
