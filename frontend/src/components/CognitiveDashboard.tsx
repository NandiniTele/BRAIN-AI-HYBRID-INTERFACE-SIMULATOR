
import { motion } from 'framer-motion';
import { Brain, Eye, Zap, HeartPulse, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ScatterChart, Scatter, ZAxis, LineChart, Line } from 'recharts';

interface CognitiveDashboardProps {
  predictions: any;
  bands: any;
  history: any[];
}

export default function CognitiveDashboard({ predictions, bands, history }: CognitiveDashboardProps) {
  const bandData = bands ? [
    { name: 'Delta', value: bands.delta, fill: '#ff2a9d' },
    { name: 'Theta', value: bands.theta, fill: '#b52aff' },
    { name: 'Alpha', value: bands.alpha, fill: '#00f0ff' },
    { name: 'Beta', value: bands.beta, fill: '#00ff9d' },
    { name: 'Gamma', value: bands.gamma, fill: '#ffea00' },
  ] : [];

  const emotionData = predictions ? [
    { x: predictions.valence, y: predictions.arousal, z: 1 }
  ] : [];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 relative flex flex-col gap-5 overflow-hidden transition-all duration-300 hover:border-[#00ff9d]/30 group"
    >
      <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#00ff9d] to-transparent opacity-50"></div>

      <div className="flex justify-between items-center">
        <h2 className="text-sm font-bold tracking-widest text-gray-400 flex items-center gap-2 font-mono" title="Machine Learning Inference Engine">
          <Brain className="w-4 h-4 text-[#00ff9d]" />
          INFERENCE ENGINE
        </h2>
        <span className="text-[10px] text-[#00ff9d] border border-[#00ff9d]/30 bg-[#00ff9d]/10 px-2 py-0.5 rounded animate-pulse" title="Model Prediction Confidence">
          CONF: {predictions?.confidence.toFixed(1)}%
        </span>
      </div>

      {predictions ? (
        <>
          {/* Gauges */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: 'FOCUS', val: predictions.focus, color: '#00f0ff', icon: <Eye className="w-3 h-3"/>, desc: 'Sustained attention level' },
              { label: 'ATTENTION', val: predictions.attention, color: '#b52aff', icon: <Zap className="w-3 h-3"/>, desc: 'Instantaneous alertness' },
              { label: 'STRESS', val: predictions.stress, color: '#ff2a9d', icon: <HeartPulse className="w-3 h-3"/>, desc: 'Cognitive load & stress' },
              { label: 'FATIGUE', val: predictions.fatigue, color: '#ffea00', icon: <Activity className="w-3 h-3"/>, desc: 'Mental exhaustion estimation' }
            ].map((gauge) => (
              <motion.div 
                whileHover={{ scale: 1.05 }}
                key={gauge.label} 
                title={gauge.desc}
                className="relative flex flex-col items-center justify-center p-2 bg-[#05050a] border border-gray-800 rounded-lg cursor-help transition-colors hover:border-[#00f0ff]/50"
              >
                <svg className="w-12 h-12 transform -rotate-90">
                  <circle cx="24" cy="24" r="20" stroke="currentColor" strokeWidth="3" fill="transparent" className="text-gray-900" />
                  <motion.circle 
                    cx="24" cy="24" r="20" 
                    stroke={gauge.color} 
                    strokeWidth="3" 
                    fill="transparent"
                    strokeDasharray="125"
                    initial={{ strokeDashoffset: 125 }}
                    animate={{ strokeDashoffset: 125 - (125 * gauge.val) / 100 }}
                    transition={{ duration: 0.5 }}
                    style={{ filter: `drop-shadow(0 0 3px ${gauge.color}80)` }}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 -mt-3 text-center">
                  <span className="text-[10px] font-bold font-mono text-white">{gauge.val.toFixed(0)}</span>
                </div>
                <div className="text-[8px] font-mono text-gray-500 mt-1 flex items-center gap-1">
                  <span style={{color: gauge.color}}>{gauge.icon}</span> {gauge.label}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Spectral Power */}
            <motion.div whileHover={{ scale: 1.02 }} className="bg-[#05050a] border border-gray-800 p-3 rounded-lg hover:border-[#b52aff]/50 transition-colors cursor-crosshair">
              <div className="text-[9px] font-mono text-gray-500 mb-2 tracking-widest flex justify-between" title="Power Spectral Density by Frequency Band">
                <span>PSD BANDS</span>
                <span className="text-[#00f0ff]">dB/Hz</span>
              </div>
              <div className="h-28">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={bandData} layout="vertical" margin={{top: 0, right: 0, left: -25, bottom: 0}}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: '#888', fontSize: 9, fontFamily: 'monospace'}} />
                    <Tooltip cursor={{fill: '#1a1a2e'}} contentStyle={{backgroundColor: '#0a0a1a', border: '1px solid #333', fontSize: '10px', fontFamily: 'monospace', borderRadius: '8px'}} />
                    <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                      {bandData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} style={{ filter: `drop-shadow(0 0 4px ${entry.fill}80)` }} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Emotion Space */}
            <motion.div whileHover={{ scale: 1.02 }} className="bg-[#05050a] border border-gray-800 p-3 rounded-lg flex flex-col hover:border-[#ffea00]/50 transition-colors cursor-crosshair">
              <div className="text-[9px] font-mono text-gray-500 mb-2 tracking-widest flex justify-between" title="Russell's Circumplex Model of Affect (Valence vs Arousal)">
                <span>VAL vs ARO</span>
                <span className="text-[#ffea00] font-bold">{predictions.emotion.toUpperCase()}</span>
              </div>
              <div className="flex-1 relative border border-gray-800/50 rounded bg-[#030308] overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center opacity-30">
                  <div className="w-full h-[1px] bg-gray-500"></div>
                  <div className="h-full w-[1px] bg-gray-500 absolute"></div>
                </div>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{top: 5, right: 5, left: -25, bottom: 0}}>
                    <XAxis type="number" dataKey="x" domain={[-1, 1]} hide />
                    <YAxis type="number" dataKey="y" domain={[-1, 1]} hide />
                    <ZAxis type="number" dataKey="z" range={[150, 150]} />
                    <Scatter data={emotionData} fill="#ffea00" animationDuration={400}>
                      {emotionData.map((_entry, index) => (
                         <Cell key={`cell-${index}`} fill="#ffea00" style={{ filter: 'drop-shadow(0 0 8px #ffea00)' }} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                <div className="absolute top-1 left-1 text-[7px] text-gray-600 font-mono">ANGRY</div>
                <div className="absolute top-1 right-1 text-[7px] text-gray-600 font-mono">EXCITED</div>
                <div className="absolute bottom-1 left-1 text-[7px] text-gray-600 font-mono">SAD</div>
                <div className="absolute bottom-1 right-1 text-[7px] text-gray-600 font-mono">RELAXED</div>
              </div>
            </motion.div>
          </div>

          {/* Cognitive Timeline */}
          <motion.div whileHover={{ scale: 1.01 }} className="bg-[#05050a] border border-gray-800 p-3 rounded-lg hover:border-gray-600 transition-colors cursor-text">
             <div className="text-[9px] font-mono text-gray-500 mb-2 tracking-widest" title="Historical progression of primary cognitive states">COGNITIVE TIMELINE (ROLLING)</div>
             <div className="h-16">
               <ResponsiveContainer width="100%" height="100%">
                 <LineChart data={history} margin={{top: 0, right: 0, left: 0, bottom: 0}}>
                   <YAxis domain={[0, 100]} hide />
                   <Tooltip contentStyle={{backgroundColor: '#0a0a1a', border: '1px solid #333', fontSize: '10px', fontFamily: 'monospace', borderRadius: '8px'}} labelStyle={{display: 'none'}} />
                   <Line type="monotone" dataKey="focus" stroke="#00f0ff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   <Line type="monotone" dataKey="attention" stroke="#b52aff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   <Line type="monotone" dataKey="stress" stroke="#ff2a9d" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   <Line type="monotone" dataKey="confidence" stroke="#ffea00" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                 </LineChart>
               </ResponsiveContainer>
             </div>
             <div className="flex gap-3 justify-center mt-1 text-[8px] font-mono">
                <span className="text-[#00f0ff]">● FOCUS</span>
                <span className="text-[#b52aff]">● ATTN</span>
                <span className="text-[#ff2a9d]">● STRESS</span>
                <span className="text-[#ffea00]">● CONF</span>
             </div>
          </motion.div>
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center min-h-[300px]">
          <div className="w-8 h-8 border-2 border-t-[#00ff9d] border-gray-800 rounded-full animate-spin"></div>
        </div>
      )}
    </motion.div>
  );
}
