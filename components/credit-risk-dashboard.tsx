"use client";

import { useMemo, useState } from "react";
import { Activity, BarChart3, BrainCircuit, FlaskConical, Gauge, LayoutDashboard, ShieldCheck } from "lucide-react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { assessRisk, generatePortfolio, modelDiagnostics, stressApplication } from "@/lib/risk/engine";
import type { ApplicationInput, PortfolioRecord, RiskResult } from "@/lib/risk/types";

type View = "overview" | "underwrite" | "portfolio" | "model" | "stress" | "method";

const initialApplication: ApplicationInput = {
  borrowerName: "Jordan Ellis",
  annualIncome: 92000,
  debtToIncome: 0.31,
  creditUtilization: 0.37,
  delinquencies24m: 0,
  inquiries6m: 1,
  oldestTradeMonths: 122,
  openAccounts: 8,
  loanAmount: 28000,
  termMonths: 36,
  employmentYears: 6.5,
  cashBufferMonths: 4.2,
  onTimePaymentRate: 0.986,
  incomeStability: 0.86,
  recentCreditGrowth: 0.12,
};

const pct = (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`;
const money = (n: number) => `$${Math.round(n).toLocaleString()}`;
const navItems = [
  ["overview", LayoutDashboard, "Overview"],
  ["underwrite", Gauge, "Underwrite"],
  ["portfolio", BarChart3, "Portfolio"],
  ["model", BrainCircuit, "Model Lab"],
  ["stress", FlaskConical, "Stress Lab"],
  ["method", ShieldCheck, "Methodology"],
] as const;

function riskBand(pd: number) {
  if (pd < 0.06) return "Prime";
  if (pd < 0.105) return "Near-prime";
  if (pd < 0.22) return "Elevated";
  return "High risk";
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="panel"><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-delta">{note}</div></div>;
}

function Overview({ portfolio }: { portfolio: PortfolioRecord[] }) {
  const approved = portfolio.filter(x => x.decision === "APPROVE").length;
  const exposure = portfolio.reduce((s, x) => s + x.ead, 0);
  const el = portfolio.reduce((s, x) => s + x.expectedLoss, 0);
  const avgPd = portfolio.reduce((s, x) => s + x.pd, 0) / portfolio.length;
  const buckets = Array.from({ length: 10 }, (_, i) => {
    const lo = i / 10, hi = (i + 1) / 10;
    const rows = portfolio.filter(x => x.pd >= lo && x.pd < hi);
    return { band: `${i * 10}-${(i + 1) * 10}%`, count: rows.length, exposure: rows.reduce((s, x) => s + x.ead, 0) / 1000 };
  });
  const sorted = [...portfolio].sort((a, b) => b.expectedLoss - a.expectedLoss).slice(0, 7);
  return <>
    <div className="grid4">
      <Metric label="Portfolio exposure" value={money(exposure)} note={`${portfolio.length} simulated applications`} />
      <Metric label="Expected loss" value={money(el)} note={`${pct(el / exposure)} of current EAD`} />
      <Metric label="Average PD" value={pct(avgPd)} note={`Risk band: ${riskBand(avgPd)}`} />
      <Metric label="Auto-approval" value={pct(approved / portfolio.length, 0)} note="Policy gate after model scoring" />
    </div>
    <div className="grid2">
      <div className="panel"><div className="panel-title"><h2>Risk distribution</h2><span className="badge">PD buckets</span></div><div className="chart"><ResponsiveContainer><AreaChart data={buckets}><defs><linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#5ee6a8" stopOpacity={0.35}/><stop offset="95%" stopColor="#5ee6a8" stopOpacity={0.02}/></linearGradient></defs><CartesianGrid stroke="#20312c" vertical={false}/><XAxis dataKey="band" stroke="#789087" fontSize={10}/><YAxis stroke="#789087" fontSize={10}/><Tooltip contentStyle={{background:"#0c1714",border:"1px solid #294139",borderRadius:8}}/><Area type="monotone" dataKey="count" stroke="#5ee6a8" fill="url(#riskFill)"/></AreaChart></ResponsiveContainer></div></div>
      <div className="panel"><div className="panel-title"><h2>Exposure by PD bucket</h2><span className="badge">$000s</span></div><div className="chart"><ResponsiveContainer><AreaChart data={buckets}><CartesianGrid stroke="#20312c" vertical={false}/><XAxis dataKey="band" stroke="#789087" fontSize={10}/><YAxis stroke="#789087" fontSize={10}/><Tooltip contentStyle={{background:"#0c1714",border:"1px solid #294139",borderRadius:8}}/><Area type="monotone" dataKey="exposure" stroke="#75a7ff" fill="#75a7ff22"/></AreaChart></ResponsiveContainer></div></div>
    </div>
    <div className="panel"><div className="panel-title"><h2>Largest expected-loss contributors</h2><span className="badge">Top 7 accounts</span></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>Score</th><th>PD</th><th>EAD</th><th>LGD</th><th>Expected loss</th><th>Decision</th></tr></thead><tbody>{sorted.map(r=><tr key={r.id}><td>{r.id}</td><td>{r.score} · {r.grade}</td><td>{pct(r.pd)}</td><td>{money(r.ead)}</td><td>{pct(r.lgd)}</td><td>{money(r.expectedLoss)}</td><td className={`decision ${r.decision.toLowerCase()}`}>{r.decision}</td></tr>)}</tbody></table></div></div>
  </>;
}

function Underwrite() {
  const [input, setInput] = useState(initialApplication);
  const [result, setResult] = useState<RiskResult>(() => assessRisk(initialApplication));
  const update = (key: keyof ApplicationInput, value: string) => setInput(prev => ({ ...prev, [key]: key === "borrowerName" ? value : Number(value) }));
  const fields: Array<[keyof ApplicationInput, string, number, number, number]> = [
    ["annualIncome","Annual income",10000,500000,1000],["loanAmount","Requested amount",1000,150000,500],["debtToIncome","Debt / income",0,0.9,.01],
    ["creditUtilization","Credit utilization",0,1.2,.01],["delinquencies24m","Delinquencies · 24m",0,10,1],["inquiries6m","Inquiries · 6m",0,12,1],
    ["oldestTradeMonths","Oldest trade · months",3,420,1],["openAccounts","Open accounts",1,35,1],["employmentYears","Employment · years",0,35,.5],
    ["cashBufferMonths","Cash buffer · months",0,18,.1],["onTimePaymentRate","On-time payment rate",.45,1,.001],["incomeStability","Income stability",0,1,.01],
    ["recentCreditGrowth","Recent credit growth",-.5,1.5,.01],["termMonths","Term · months",6,84,6],
  ];
  return <div className="grid2">
    <div className="panel"><div className="panel-title"><h2>Application vector</h2><span className="badge">14 signals + derived features</span></div><div className="field" style={{marginBottom:12}}><label>Borrower / reference</label><input value={input.borrowerName} onChange={e=>update("borrowerName",e.target.value)}/></div><div className="form-grid">{fields.map(([key,label,min,max,step])=><div className="field" key={key}><label>{label}</label><input type="number" min={min} max={max} step={step} value={input[key] as number} onChange={e=>update(key,e.target.value)}/></div>)}</div><div className="actions"><button className="primary" onClick={()=>setResult(assessRisk(input))}>Run decision engine</button><button className="secondary" onClick={()=>{setInput(initialApplication);setResult(assessRisk(initialApplication));}}>Reset</button></div></div>
    <div className="panel"><div className="panel-title"><h2>Decision output</h2><span className={`decision ${result.decision.toLowerCase()}`}>{result.decision}</span></div><div className="hero-result"><div className="score-ring"><div><strong>{result.score}</strong><span>CRIX score · {result.grade}</span></div></div><div><div className="result-grid"><div className="result-cell"><span className="muted">Calibrated PD</span><strong>{pct(result.pd,2)}</strong></div><div className="result-cell"><span className="muted">Expected loss</span><strong>{money(result.expectedLoss)}</strong></div><div className="result-cell"><span className="muted">Confidence</span><strong>{pct(result.confidence)}</strong></div><div className="result-cell"><span className="muted">LGD</span><strong>{pct(result.lgd)}</strong></div><div className="result-cell"><span className="muted">Suggested APR</span><strong>{result.apr.toFixed(2)}%</strong></div><div className="result-cell"><span className="muted">Model delta</span><strong>{pct(result.disagreement,2)}</strong></div></div></div></div><div className="reasons"><div className="metric-label">Local sensitivity / reason codes</div>{result.reasons.map(r=><div className="reason" key={r.feature}><span>{r.label}</span><span className={r.direction === "risk-up" ? "decline" : "approve"}>{r.impact >= 0 ? "+" : ""}{(r.impact*100).toFixed(2)} pp PD</span></div>)}</div>{result.outOfDistribution.length > 0 && <div className="callout" style={{marginTop:14}}>Out-of-distribution warning: {result.outOfDistribution.join(", ")}. Automated decisions should be suppressed for values outside the model development envelope.</div>}<div className="callout" style={{marginTop:14}}>Champion {result.modelVersion} is compared against an interpretable logistic challenger. Material disagreement reduces confidence before the separate policy layer assigns approve, review or decline.</div></div>
  </div>;
}

function Portfolio({ portfolio }: { portfolio: PortfolioRecord[] }) {
  const [filter, setFilter] = useState<"ALL"|"APPROVE"|"REVIEW"|"DECLINE">("ALL");
  const rows = portfolio.filter(x => filter === "ALL" || x.decision === filter).slice(0, 80);
  return <div className="panel"><div className="panel-title"><h2>Portfolio explorer</h2><div className="actions" style={{margin:0}}>{(["ALL","APPROVE","REVIEW","DECLINE"] as const).map(x=><button key={x} className={filter===x?"primary":"secondary"} onClick={()=>setFilter(x)}>{x}</button>)}</div></div><div className="table-wrap"><table><thead><tr><th>Applicant</th><th>Income</th><th>Request</th><th>DTI</th><th>Util.</th><th>Score</th><th>PD</th><th>EL</th><th>Decision</th></tr></thead><tbody>{rows.map(r=><tr key={r.id}><td>{r.borrowerName}</td><td>{money(r.annualIncome)}</td><td>{money(r.loanAmount)}</td><td>{pct(r.debtToIncome)}</td><td>{pct(r.creditUtilization)}</td><td>{r.score}</td><td>{pct(r.pd)}</td><td>{money(r.expectedLoss)}</td><td className={`decision ${r.decision.toLowerCase()}`}>{r.decision}</td></tr>)}</tbody></table></div></div>;
}

function ModelLab() {
  const d = modelDiagnostics;
  return <>
    <div className="panel" style={{marginBottom:12}}><div className="panel-title"><h2>{d.name} · model validation</h2><span className="badge">held-out n={d.metrics.testSamples.toLocaleString()}</span></div><div className="model-metrics"><div className="result-cell"><span className="muted">ROC-AUC</span><strong>{d.metrics.auc.toFixed(3)}</strong></div><div className="result-cell"><span className="muted">KS statistic</span><strong>{d.metrics.ks.toFixed(3)}</strong></div><div className="result-cell"><span className="muted">Brier score</span><strong>{d.metrics.brier.toFixed(3)}</strong></div><div className="result-cell"><span className="muted">Log loss</span><strong>{d.metrics.logLoss.toFixed(3)}</strong></div></div></div>
    <div className="grid2"><div className="panel"><div className="panel-title"><h2>Calibration / reliability</h2><span className="badge">predicted vs observed</span></div><div className="chart"><ResponsiveContainer><LineChart data={d.diagnostics.calibration}><CartesianGrid stroke="#20312c"/><XAxis dataKey="predicted" stroke="#789087" tickFormatter={(v)=>pct(v,0)} fontSize={10}/><YAxis stroke="#789087" tickFormatter={(v)=>pct(v,0)} fontSize={10}/><Tooltip contentStyle={{background:"#0c1714",border:"1px solid #294139",borderRadius:8}} formatter={(v)=>pct(Number(v),2)}/><Line type="monotone" dataKey="observed" stroke="#5ee6a8" strokeWidth={2}/><Line type="monotone" dataKey="predicted" stroke="#75a7ff" strokeDasharray="4 4"/></LineChart></ResponsiveContainer></div></div><div className="panel"><div className="panel-title"><h2>Global feature importance</h2><span className="badge">normalized gain</span></div><div className="bar-list">{d.diagnostics.featureImportance.slice(0,10).map(x=><div className="bar-row" key={x.feature}><span>{x.feature}</span><div className="bar-track"><div className="bar-fill" style={{width:`${Math.max(3,x.gain*100)}%`}}/></div><span>{pct(x.gain,0)}</span></div>)}</div></div></div>
    <div className="callout">The champion is a shallow monotonic gradient-boosted tree ensemble. Monotonic constraints encode domain priors for variables such as DTI, utilization, delinquency burden and payment performance; Platt calibration is fit on a separate calibration split so the output can be treated as a probability rather than only a ranking score.</div>
  </>;
}

function StressLab({ portfolio }: { portfolio: PortfolioRecord[] }) {
  const baseEl = portfolio.reduce((s,x)=>s+x.expectedLoss,0);
  const mild = portfolio.map(x=>stressApplication(x,"mild").result);
  const severe = portfolio.map(x=>stressApplication(x,"severe").result);
  const summarize=(rows:RiskResult[])=>({el:rows.reduce((s,x)=>s+x.expectedLoss,0),pd:rows.reduce((s,x)=>s+x.pd,0)/rows.length,decline:rows.filter(x=>x.decision==="DECLINE").length/rows.length});
  const b={el:baseEl,pd:portfolio.reduce((s,x)=>s+x.pd,0)/portfolio.length,decline:portfolio.filter(x=>x.decision==="DECLINE").length/portfolio.length}, m=summarize(mild), s=summarize(severe);
  const cards=[["Base case",b],["Mild recession",m],["Severe recession",s]] as const;
  return <><div className="scenario">{cards.map(([name,x])=><div className="scenario-card" key={name}><h3>{name}</h3><div className="scenario-big">{money(x.el)}</div><span className="muted">portfolio expected loss</span><div className="scenario-grid"><div>Average PD<strong>{pct(x.pd)}</strong></div><div>Decline rate<strong>{pct(x.decline,0)}</strong></div><div>EL uplift<strong>{pct(x.el/baseEl-1)}</strong></div><div>Scenario<strong>{name === "Base case" ? "Observed" : "Macro shock"}</strong></div></div></div>)}</div><div className="callout" style={{marginTop:12}}>Stress shocks reduce income and liquidity while increasing DTI, utilization and recent credit growth. Every account is re-scored through the same champion, challenger and policy layers; this is a sensitivity engine, not a claim of macroeconomic forecasting.</div></>;
}

function Methodology() {
  return <div className="method"><div className="panel"><h2>Architecture</h2><p>CRIX is intentionally infrastructure-light. Model development happens offline; the trained tree artifact, calibration coefficients, reference vector and diagnostics are committed with the application. The browser runs deterministic inference directly, so the live demo has no database, Redis cluster, Python server or paid API dependency.</p></div><div className="panel"><h2>Risk stack</h2><ul><li><span className="codepill">PD</span> — calibrated 12-month probability of default from the monotonic champion.</li><li><span className="codepill">LGD</span> — scenario-aware unsecured loss severity estimate.</li><li><span className="codepill">EAD</span> — requested funded exposure.</li><li><span className="codepill">EL = PD × LGD × EAD</span> — account expected loss.</li><li><span className="codepill">CRIX score</span> — probability-to-odds scaling on a 300–850 range with points-to-double-odds behaviour.</li></ul></div><div className="panel"><h2>Model governance</h2><p>The model output is separated from policy. A challenger model, out-of-distribution checks, local sensitivity reason codes and stress tests make failure modes observable. The training pipeline is reproducible and seeded. This repository is an engineering/research demonstration and must not be used for real consumer lending without representative data, legal review, fairness testing, independent validation and production monitoring.</p></div></div>;
}

export function CreditRiskDashboard() {
  const [view,setView]=useState<View>("overview");
  const portfolio=useMemo(()=>generatePortfolio(),[]);
  return <div className="shell"><aside className="sidebar"><div className="brand"><div className="brandmark">CΞ</div><div><strong>CRIX</strong><small>Credit Risk Intelligence</small></div></div><nav className="nav">{navItems.map(([id,Icon,label])=><button key={id} className={view===id?"active":""} onClick={()=>setView(id)}><Icon size={17}/><span>{label}</span></button>)}</nav><div className="sidebar-foot"><Activity size={15} style={{marginBottom:7}}/><br/>Model artifact loaded locally.<br/>No external runtime dependencies.</div></aside><main className="main"><header className="topbar"><div><div className="eyebrow">Risk decisioning laboratory</div><h1>{navItems.find(x=>x[0]===view)?.[2]}</h1><p className="sub">Explainable underwriting, calibrated probability of default, portfolio loss intelligence and model-risk diagnostics in one self-contained system.</p></div><div className="status"><span className="dot"/>CRIX-MonoBoost v1.0 · healthy</div></header>{view==="overview"&&<Overview portfolio={portfolio}/>} {view==="underwrite"&&<Underwrite/>} {view==="portfolio"&&<Portfolio portfolio={portfolio}/>} {view==="model"&&<ModelLab/>} {view==="stress"&&<StressLab portfolio={portfolio}/>} {view==="method"&&<Methodology/>}</main></div>;
}
