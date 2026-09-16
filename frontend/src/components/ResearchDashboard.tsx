import { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Activity, Info, BarChart2, CheckCircle, AlertTriangle, FileText, Download, BrainCircuit, Shield, Bot, Layers } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ResearchDashboardProps {
  xai: any;
  signalQuality: any;
  researchData: any;
  training: any;
  modelStats: any;
  modelComparison: any;
  datasetMeta: any;
  sessionAnalytics: any;
  system: any;
  regionActivity: any;
  channelAnalytics: any;
  aiAssistant: string;
}

export default function ResearchDashboard({
  xai, signalQuality, researchData, training, modelStats, modelComparison, datasetMeta, sessionAnalytics, system, regionActivity, channelAnalytics, aiAssistant
}: ResearchDashboardProps) {
  
  const [activeTab, setActiveTab] = useState('OVERVIEW');

  if (!xai || !modelStats) return null;

  const handleExport = (type: string) => {
    if (type === 'pdf') {
      window.print();
      return;
    }

    let content = '';
    let mimeType = 'text/plain';

    if (type === 'csv') {
      const headers = Object.keys(sessionAnalytics).join(',');
      const values = Object.values(sessionAnalytics).join(',');
      content = `${headers}\n${values}`;
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(sessionAnalytics, null, 2);
      mimeType = 'application/json';
    }

    const filename = `session_report_${Date.now()}.${type}`;
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const el = document.createElement('a');
    el.setAttribute('href', url);
    el.setAttribute('download', filename);
    el.style.display = 'none';
    document.body.appendChild(el);
    el.click();
    document.body.removeChild(el);
    URL.revokeObjectURL(url);
  };

  const handleSnapshot = (action: 'save' | 'load') => {
    if (action === 'save') {
      localStorage.setItem('neural_snapshot', JSON.stringify(sessionAnalytics));
      alert('Session snapshot encrypted and saved to local storage.');
    } else {
      const data = localStorage.getItem('neural_snapshot');
      if (data) alert('Previous session snapshot loaded successfully.');
      else alert('No snapshot found.');
    }
  };

  const featureImportanceData = xai.feature_importance 
    ? Object.entries(xai.feature_importance).map(([key, val]) => ({ name: key, value: Number(val) * 100 }))
    : [];

  const regionData = regionActivity 
    ? Object.entries(regionActivity).map(([key, val]) => ({ name: key, value: Number(val) }))
    : [];

  const formatTensorData = (tensor: any) => {
    if (!tensor) return 'Waiting for tensor...';
    try {
      let parsed = typeof tensor === 'string' ? JSON.parse(tensor) : tensor;
      let str = JSON.stringify(parsed, (_key, val) => 
        typeof val === 'number' ? Number(val.toFixed(4)) : val
      );
      
      str = str.replace(/,/g, ', ');
      
      if (str.includes('], [')) {
         str = str.replace(/\], \[/g, '],\n  [').replace(/^\[\[/, '[\n  [').replace(/\]\]$/, ']\n]');
      } else if (Array.isArray(parsed) && !Array.isArray(parsed[0])) {
         let chunks = [];
         for (let i = 0; i < parsed.length; i += 10) {
             chunks.push('  ' + parsed.slice(i, i + 10).map((v: any) => v.toFixed ? v.toFixed(4) : v).join(', '));
         }
         str = '[\n' + chunks.join(',\n') + '\n]';
      }
      
      return str.length > 5000 ? str.substring(0, 5000) + '\n\n... [TRUNCATED - EXCEEDS MAX DISPLAY LENGTH]' : str;
    } catch {
      const fallback = JSON.stringify(tensor);
      return fallback.length > 1000 ? fallback.substring(0, 1000) + '...' : fallback;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="flex flex-col gap-4 mt-4">
      
      {/* TABS NAVIGATION */}
      <div className="flex gap-1 border-b border-gray-800 pb-0 overflow-x-auto hide-scrollbar">
        {['OVERVIEW', 'PIPELINE & TENSORS', 'REGION & CHANNEL ANALYTICS', 'AI ASSISTANT & EXPERIMENTS'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`relative px-4 py-2.5 text-[10px] font-mono transition-all whitespace-nowrap ${
              activeTab === tab
                ? 'text-[#00f0ff]'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {tab}
            {activeTab === tab && (
              <motion.div
                layoutId="tab-underline"
                className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r from-[#00f0ff] to-[#b52aff]"
                style={{ boxShadow: '0 0 8px #00f0ff' }}
              />
            )}
          </button>
        ))}
      </div>

      {activeTab === 'OVERVIEW' && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* 1. XAI & AI Model Info */}
          <div className="glass-panel p-4 rounded-xl border border-[#b52aff]/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-[#b52aff] flex items-center gap-2 tracking-widest"><Info className="w-4 h-4"/> EXPLAINABLE AI (XAI)</h3>
            
            <div className="bg-[#05050a] p-3 rounded border border-gray-800">
               <div className="text-[10px] text-gray-500 mb-2">FEATURE ATTRIBUTION (SHAP/LIME proxy)</div>
               <div className="h-32">
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featureImportanceData} layout="vertical" margin={{top: 0, right: 0, left: 0, bottom: 0}}>
                      <XAxis type="number" hide />
                      <YAxis dataKey="name" type="category" width={50} tick={{fill: '#888', fontSize: 10}} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{backgroundColor: '#0a0a1a', border: '1px solid #333', fontSize: '10px'}} />
                      <Bar dataKey="value" fill="#b52aff" radius={[0, 4, 4, 0]} />
                    </BarChart>
                 </ResponsiveContainer>
               </div>
               <div className="mt-2 text-[10px] text-gray-400">
                 <span className="text-white">Primary Driver:</span> {xai.primary_driver}
               </div>
            </div>

            <div className="bg-[#05050a] p-3 rounded border border-gray-800 flex-1">
              <div className="text-[10px] text-gray-500 mb-2">ACTIVE MODEL ARCHITECTURE</div>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <div className="text-gray-400">Name:</div><div className="text-white text-right">{modelStats.architecture}</div>
                <div className="text-gray-400">Framework:</div><div className="text-[#ff2a9d] text-right font-bold">{modelStats.framework}</div>
                <div className="text-gray-400">Device:</div><div className="text-[#00ff9d] text-right font-bold">{modelStats.device}</div>
                <div className="text-gray-400">Parameters:</div><div className="text-white text-right">{modelStats.parameters}</div>
                <div className="text-gray-400">ROC AUC:</div><div className="text-white text-right">{modelStats.roc_auc}</div>
                <div className="text-gray-400">Status:</div><div className="text-green-400 text-right animate-pulse">{modelStats.status}</div>
              </div>
            </div>
          </div>

          {/* 2. Model Comparison & Training */}
          <div className="glass-panel p-4 rounded-xl border border-[#00f0ff]/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-[#00f0ff] flex items-center gap-2 tracking-widest"><BarChart2 className="w-4 h-4"/> MODEL ANALYTICS</h3>
            
            <div className="bg-[#05050a] p-3 rounded border border-gray-800">
               <div className="text-[10px] text-gray-500 mb-2">ARCHITECTURE COMPARISON</div>
               <div className="h-32">
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={modelComparison} margin={{top: 0, right: 0, left: -20, bottom: 0}}>
                      <XAxis dataKey="name" tick={{fill: '#888', fontSize: 9}} axisLine={false} tickLine={false} />
                      <YAxis domain={[80, 100]} hide />
                      <Tooltip contentStyle={{backgroundColor: '#0a0a1a', border: '1px solid #333', fontSize: '10px'}} />
                      <Bar dataKey="acc" radius={[4, 4, 0, 0]}>
                        {modelComparison.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={entry.name === 'HybridBCINet' ? '#00f0ff' : '#333'} />
                        ))}
                      </Bar>
                    </BarChart>
                 </ResponsiveContainer>
               </div>
            </div>

            <div className="bg-[#05050a] p-3 rounded border border-gray-800 flex-1">
               <div className="text-[10px] text-gray-500 mb-2 flex justify-between">
                 <span>TRAINING DASHBOARD</span>
                 <span>Epoch {modelStats.epoch}</span>
               </div>
               {training?.is_training ? (
                 <div className="text-center">
                   <div className="text-xs text-[#00ff9d] mb-1 animate-pulse">TRAINING IN PROGRESS...</div>
                   <div className="w-full bg-gray-800 rounded h-2 mb-2"><div className="bg-[#00ff9d] h-2 rounded" style={{width: `${training.progress}%`}}></div></div>
                 </div>
               ) : (
                 <div className="grid grid-cols-2 gap-2 text-[10px]">
                   <div className="text-gray-400">Validation Acc:</div><div className="text-white text-right">{modelStats.accuracy}%</div>
                   <div className="text-gray-400">Precision:</div><div className="text-white text-right">{modelStats.precision}%</div>
                   <div className="text-gray-400">Recall:</div><div className="text-white text-right">{modelStats.recall}%</div>
                   <div className="text-gray-400">F1-Score:</div><div className="text-[#ffea00] text-right font-bold">{modelStats.f1_score}%</div>
                   <div className="text-gray-400">Prediction Confidence:</div><div className="text-white text-right">{modelStats.cross_val_score}%</div>
                 </div>
               )}
            </div>
          </div>

          {/* 3. Data & Signal Quality */}
          <div className="glass-panel p-4 rounded-xl border border-[#00ff9d]/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-[#00ff9d] flex items-center gap-2 tracking-widest"><Activity className="w-4 h-4"/> SIGNAL & DATA</h3>
            
            <div className="bg-[#05050a] p-3 rounded border border-gray-800">
               <div className="text-[10px] text-gray-500 mb-2">SIGNAL QUALITY MONITOR</div>
               <div className="flex items-center justify-between mb-2">
                 <span className="text-[10px] text-gray-400">Overall Quality</span>
                 <span className="text-xs font-bold" style={{color:'#00ff9d', textShadow:'0 0 8px #00ff9d'}}>{signalQuality.quality_percent.toFixed(1)}%</span>
               </div>
               <div className="w-full bg-gray-900 rounded-full h-2 mb-3 overflow-hidden">
                 <motion.div
                   initial={{ width: 0 }}
                   animate={{ width: `${signalQuality.quality_percent}%` }}
                   transition={{ duration: 1, ease: 'easeOut' }}
                   className="h-2 rounded-full"
                   style={{ background: 'linear-gradient(90deg,#00ff9d,#00f0ff)', boxShadow: '0 0 10px #00ff9d88' }}
                 />
               </div>
               
               <div className="grid grid-cols-2 gap-2 text-[10px]">
                 <div className="text-gray-400">SNR (dB):</div><div className="text-white text-right">{signalQuality.snr_db.toFixed(2)}</div>
                 <div className="text-gray-400">Noise Level:</div><div className="text-white text-right">{signalQuality.noise_level.toFixed(3)}</div>
                 <div className="text-gray-400">Artifacts:</div>
                 <div className="text-right">
                   {signalQuality.artifact_detected ? <span className="text-red-500 flex items-center justify-end gap-1"><AlertTriangle className="w-3 h-3"/> DETECTED</span> : <span className="text-green-500 flex items-center justify-end gap-1"><CheckCircle className="w-3 h-3"/> CLEAN</span>}
                 </div>
               </div>
            </div>

            <div className="bg-[#05050a] p-3 rounded border border-gray-800 flex-1">
               <div className="text-[10px] text-gray-500 mb-2">DATASET ANALYTICS ({datasetMeta.current})</div>
               {datasetMeta[datasetMeta.current] && (
                 <div className="grid grid-cols-2 gap-2 text-[10px]">
                   <div className="text-gray-400">Subjects:</div><div className="text-white text-right">{datasetMeta[datasetMeta.current].subjects}</div>
                   <div className="text-gray-400">Channels:</div><div className="text-white text-right">{datasetMeta[datasetMeta.current].channels}</div>
                   <div className="text-gray-400">Sampling Rate:</div><div className="text-white text-right">{datasetMeta[datasetMeta.current].fs} Hz</div>
                   <div className="text-gray-400">Duration:</div><div className="text-white text-right">{datasetMeta[datasetMeta.current].duration}</div>
                 </div>
               )}
            </div>
          </div>

          {/* 4. Session & System */}
          <div className="glass-panel p-4 rounded-xl border border-[#ffea00]/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-[#ffea00] flex items-center gap-2 tracking-widest"><Database className="w-4 h-4"/> SESSION & SYSTEM</h3>
            
            <div className="bg-[#05050a] p-3 rounded border border-gray-800">
               <div className="text-[10px] text-gray-500 mb-2">ADVANCED SESSION ANALYTICS</div>
               <div className="grid grid-cols-2 gap-2 text-[10px]">
                 <div className="text-gray-400">Session ID:</div><div className="text-white text-right truncate" title={sessionAnalytics.id}>{sessionAnalytics.id}</div>
                 <div className="text-gray-400">Total Inferences:</div><div className="text-white text-right">{sessionAnalytics.total_predictions}</div>
                 <div className="text-gray-400">Avg Focus:</div><div className="text-white text-right">{sessionAnalytics.avg_focus.toFixed(1)}</div>
                 <div className="text-gray-400">Avg Stress:</div><div className="text-white text-right">{sessionAnalytics.avg_stress.toFixed(1)}</div>
                 <div className="text-gray-400">Avg Confidence:</div><div className="text-[#00ff9d] text-right">{sessionAnalytics.avg_confidence.toFixed(1)}%</div>
                 <div className="text-gray-400">Inference Latency:</div><div className="text-white text-right">{system.inference_latency ? system.inference_latency.toFixed(1) : 0.0} ms</div>
                 <div className="text-gray-400">Dominant Emotion:</div><div className="text-[#ffea00] text-right font-bold">{sessionAnalytics.dominant_emotion.toUpperCase()}</div>
               </div>
            </div>

            <div className="bg-[#05050a] p-3 rounded border border-gray-800 flex-1 flex flex-col justify-between">
               <div>
                 <div className="text-[10px] text-gray-500 mb-3">SYSTEM PERFORMANCE</div>
                 {[{ label:'CPU', value: system.cpu_usage, color:'#00f0ff' }, { label:'GPU', value: system.gpu_usage, color:'#b52aff' }, { label:'RAM', value: system.ram_usage, color:'#ffea00' }].map(g => (
                   <div key={g.label} className="mb-2">
                     <div className="flex justify-between text-[10px] mb-1">
                       <span className="text-gray-400">{g.label} Usage</span>
                       <span style={{color: g.color}}>{g.value.toFixed(1)}%</span>
                     </div>
                     <div className="w-full bg-gray-900 rounded-full h-1.5 overflow-hidden">
                       <motion.div
                         initial={{ width: 0 }}
                         animate={{ width: `${g.value}%` }}
                         transition={{ duration: 0.8, ease: 'easeOut' }}
                         className="h-1.5 rounded-full"
                         style={{ background: g.color, boxShadow: `0 0 6px ${g.color}88` }}
                       />
                     </div>
                   </div>
                 ))}
                 <div className="flex justify-between text-[10px] mt-2">
                   <span className="text-gray-400">Database</span>
                   <span className="text-[#00ff9d] font-bold" style={{textShadow:'0 0 6px #00ff9d'}}>{system.db_health}</span>
                 </div>
               </div>
               
               <div className="flex gap-2 mt-3">
                  <button onClick={() => handleExport('pdf')} className="flex-1 text-white text-[10px] py-1.5 rounded flex items-center justify-center gap-1 transition-all border border-gray-600 hover:border-[#00f0ff] hover:text-[#00f0ff]" style={{background:'linear-gradient(135deg,#1a1a2e,#0d0d1a)'}}>
                    <FileText className="w-3 h-3" /> PDF REPORT
                  </button>
                  <button onClick={() => handleExport('csv')} className="flex-1 text-white text-[10px] py-1.5 rounded flex items-center justify-center gap-1 transition-all border border-gray-600 hover:border-[#b52aff] hover:text-[#b52aff]" style={{background:'linear-gradient(135deg,#1a1a2e,#0d0d1a)'}}>
                    <Download className="w-3 h-3" /> EXPORT
                  </button>
               </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'PIPELINE & TENSORS' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="glass-panel p-4 rounded-xl border border-gray-700 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2 tracking-widest"><Layers className="w-4 h-4"/> SIGNAL PROCESSING PIPELINE</h3>
            <div className="bg-[#05050a] p-4 rounded border border-gray-800 flex-1 flex flex-col justify-between items-center text-[10px] font-mono text-[#00f0ff] relative">
              <div className="absolute top-0 bottom-0 w-[1px] bg-gradient-to-b from-[#00f0ff33] to-[#b52aff33] left-1/2 transform -translate-x-1/2 z-0" />
              {['Raw EEG Buffer', 'Band-pass Filter (0.5–45Hz)', 'Notch Filter (50Hz)', 'ICA Artifact Removal', 'PSD / Feature Extraction', 'PyTorch Tensor (1×19×27)', 'Hybrid CNN + Transformer', 'Softmax Prediction'].map((step, idx, arr) => (
                <div key={idx} className="flex items-center gap-3 z-10 w-full">
                  <div className="flex-1 flex justify-end">
                    <span className="text-[9px] text-gray-600 font-mono">{String(idx + 1).padStart(2,'0')}</span>
                  </div>
                  <div
                    className="flex-none px-4 py-1.5 rounded text-center w-2/3 text-[10px] border transition-all"
                    style={{
                      background: idx === arr.length - 1 ? 'linear-gradient(135deg,#00f0ff22,#b52aff22)' : '#050508',
                      borderColor: idx === arr.length - 1 ? '#00f0ff' : '#2a2a3a',
                      color: idx === arr.length - 1 ? '#00f0ff' : '#aaa',
                      boxShadow: idx === arr.length - 1 ? '0 0 10px #00f0ff44' : 'none'
                    }}
                  >
                    {step}
                  </div>
                  <div className="flex-1">
                    {idx < arr.length - 1 && <span className="w-1.5 h-1.5 rounded-full bg-gray-700 inline-block" />}
                    {idx === arr.length - 1 && <span className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] inline-block animate-pulse" style={{boxShadow:'0 0 6px #00f0ff'}} />}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="glass-panel p-4 rounded-xl border border-[#ff2a9d]/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-[#ff2a9d] flex items-center gap-2 tracking-widest"><BrainCircuit className="w-4 h-4"/> TENSOR INSPECTOR & ACTIVATIONS</h3>
            <div className="bg-[#05050a] p-3 rounded border border-gray-800 font-mono text-[9px] text-[#00ff9d] overflow-y-auto max-h-[400px] flex flex-col gap-4">
              
              <div>
                <div className="text-gray-400 mb-1 font-bold flex justify-between">
                  <span>1. FEATURE VECTOR TENSOR</span>
                  <span className="text-[#b52aff]">Shape: 1x19x27</span>
                </div>
                <div className="bg-[#020204] p-2 rounded border border-gray-900 overflow-x-auto whitespace-pre custom-scrollbar">
                  {formatTensorData(researchData.feature_vector)}
                </div>
              </div>
              
              <div>
                <div className="text-gray-400 mb-1 font-bold flex justify-between">
                  <span>2. CONV1D ACTIVATIONS</span>
                  <span className="text-[#b52aff]">Layer 2</span>
                </div>
                <div className="bg-[#020204] p-2 rounded border border-gray-900 overflow-x-auto whitespace-pre custom-scrollbar">
                  {formatTensorData(researchData.activations)}
                </div>
              </div>
              
              <div>
                <div className="text-gray-400 mb-1 font-bold flex justify-between">
                  <span>3. RAW EEG SAMPLE</span>
                  <span className="text-[#b52aff]">Last 50 pts</span>
                </div>
                <div className="bg-[#020204] p-2 rounded border border-gray-900 overflow-x-auto whitespace-pre custom-scrollbar text-[#ffea00]">
                  {formatTensorData(researchData.raw_eeg)}
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

      {activeTab === 'REGION & CHANNEL ANALYTICS' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="glass-panel p-4 rounded-xl border border-gray-700 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2 tracking-widest"><BrainCircuit className="w-4 h-4"/> BRAIN REGION HEAT MAP</h3>
            <div className="bg-[#05050a] p-4 rounded border border-gray-800 flex-1 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regionData}>
                  <XAxis dataKey="name" tick={{fill: '#888', fontSize: 10}} axisLine={false} tickLine={false} />
                  <YAxis hide domain={[0, 100]} />
                  <Tooltip contentStyle={{backgroundColor: '#0a0a1a', border: '1px solid #333', fontSize: '10px'}} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                     {regionData.map((_entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={['#00f0ff', '#b52aff', '#ff2a9d', '#ffea00', '#00ff9d'][index % 5]} />
                     ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="glass-panel p-4 rounded-xl border border-gray-700 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2 tracking-widest"><Activity className="w-4 h-4"/> CHANNEL ANALYTICS (10-20 SYSTEM)</h3>
            <div className="bg-[#05050a] rounded border border-gray-800 flex-1 h-64 overflow-y-auto hide-scrollbar">
              <table className="w-full text-[10px] text-left">
                <thead className="sticky top-0 bg-gray-900 border-b border-gray-800">
                  <tr>
                    <th className="p-2 text-gray-400">CH</th>
                    <th className="p-2 text-gray-400">POWER</th>
                    <th className="p-2 text-gray-400">NOISE</th>
                    <th className="p-2 text-gray-400">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {channelAnalytics?.map((ch: any) => (
                    <tr key={ch.name} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                      <td className="p-2 font-bold text-gray-300">{ch.name}</td>
                      <td className="p-2 text-[#00f0ff]">{ch.power.toFixed(3)}</td>
                      <td className="p-2 text-gray-500">{ch.noise.toFixed(3)}</td>
                      <td className="p-2">
                        {ch.artifact ? <span className="text-red-500 font-bold">ARTIFACT</span> : <span className="text-green-500">CLEAN</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'AI ASSISTANT & EXPERIMENTS' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="glass-panel p-4 rounded-xl border border-blue-500/50 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-blue-400 flex items-center gap-2 tracking-widest"><Bot className="w-4 h-4"/> AI RESEARCH ASSISTANT</h3>
            <div className="bg-[#05050a] p-4 rounded border border-gray-800 flex-1 flex flex-col">
               <div className="flex items-center gap-3 mb-4">
                 <div className="w-10 h-10 rounded-full bg-blue-900/50 border border-blue-500 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-blue-400" />
                 </div>
                 <div>
                   <div className="text-xs text-white font-bold">Project Neural-Link AI</div>
                   <div className="text-[10px] text-gray-500">System Monitoring Agent</div>
                 </div>
               </div>
               <div className="bg-blue-950/20 border border-blue-900/50 p-4 rounded-lg text-xs text-gray-300 font-mono leading-relaxed">
                 {aiAssistant}
               </div>
            </div>
          </div>

          <div className="glass-panel p-4 rounded-xl border border-gray-700 flex flex-col gap-4">
            <h3 className="text-xs font-bold text-gray-300 flex items-center gap-2 tracking-widest"><Shield className="w-4 h-4"/> EXPERIMENT MANAGER & SECURITY</h3>
            <div className="bg-[#05050a] p-4 rounded border border-gray-800 flex-1 flex flex-col gap-4">
               
               <div className="flex gap-2">
                 <button onClick={() => handleSnapshot('save')} className="flex-1 text-[10px] py-2 rounded flex items-center justify-center gap-2 transition-all border border-[#00ff9d]/40 text-[#00ff9d] hover:border-[#00ff9d] hover:shadow-[0_0_12px_#00ff9d44]" style={{background:'linear-gradient(135deg,#00ff9d0a,#05050a)'}}>
                    SAVE SESSION SNAPSHOT
                 </button>
                 <button onClick={() => handleSnapshot('load')} className="flex-1 text-[10px] py-2 rounded flex items-center justify-center gap-2 transition-all border border-[#b52aff]/40 text-[#b52aff] hover:border-[#b52aff] hover:shadow-[0_0_12px_#b52aff44]" style={{background:'linear-gradient(135deg,#b52aff0a,#05050a)'}}>
                    LOAD PREVIOUS SESSION
                 </button>
               </div>

               <div className="text-[10px] text-gray-500 mt-2">SECURITY PROTOCOL</div>
               <div className="grid grid-cols-2 gap-2 text-[10px]">
                 <div className="text-gray-400">Auth Method:</div><div className="text-[#00ff9d] text-right font-bold">JWT TOKEN</div>
                 <div className="text-gray-400">Current Role:</div><div className="text-[#b52aff] text-right font-bold">LEAD RESEARCHER</div>
                 <div className="text-gray-400">Audit Logging:</div><div className="text-green-500 text-right">ENABLED (SQLite)</div>
               </div>
            </div>
          </div>
        </div>
      )}

    </motion.div>
  );
}
