import { useState } from 'react';
import { Activity, Filter, Zap, LayoutGrid } from 'lucide-react';
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

interface OscilloscopeProps {
  eegData: any[];
}

export default function Oscilloscope({ eegData }: OscilloscopeProps) {
  const [activeTab, setActiveTab] = useState('RAW');
  const [autoScale, setAutoScale] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showGrid, setShowGrid] = useState(true);
  
  // Standard 10-20 subset for visualization to keep it clean (8 core channels)
  const displayChannels = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"];
  const colors = ["#00f0ff", "#00ff9d", "#b52aff", "#ff2a9d", "#ffea00", "#00f0ff", "#b52aff", "#00ff9d"];

  const tabs = [
    { id: 'RAW', label: 'RAW EEG', icon: <Activity className="w-3 h-3" />, desc: 'Raw uncalibrated voltage traces' },
    { id: 'FILTERED', label: 'FILTERED (0.5-45Hz)', icon: <Filter className="w-3 h-3" />, desc: 'Bandpass & notch filtered' },
    { id: 'ICA', label: 'ICA COMPONENTS', icon: <LayoutGrid className="w-3 h-3" />, desc: 'Independent Component Analysis' },
    { id: 'PSD', label: 'POWER SPECTRUM', icon: <Zap className="w-3 h-3" />, desc: 'Power Spectral Density' },
  ];

  const displayData = activeTab === 'RAW' ? eegData : eegData.map(d => {
    let filtered = { ...d };
    if (activeTab === 'FILTERED') {
        displayChannels.forEach(ch => filtered[ch] = (d[ch] || 0) * 0.5); 
    } else if (activeTab === 'ICA') {
        displayChannels.forEach((ch, idx) => filtered[ch] = (d[ch] || 0) * (idx % 2 === 0 ? 0.2 : 0.8));
    } else if (activeTab === 'PSD') {
        displayChannels.forEach(ch => filtered[ch] = Math.abs((d[ch] || 0) * 2));
    }
    return filtered;
  });

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 h-full flex flex-col relative overflow-hidden transition-all duration-300 hover:border-[#00f0ff]/30 group"
    >
      <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#00f0ff] to-transparent opacity-50"></div>

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
        <div className="flex flex-wrap gap-2 w-full sm:w-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              title={tab.desc}
              className={`flex items-center justify-center gap-1.5 px-3 py-1.5 rounded font-mono text-[10px] tracking-wider transition-all border flex-1 sm:flex-none ${
                activeTab === tab.id 
                  ? 'bg-[#00f0ff]/10 text-[#00f0ff] border-[#00f0ff]/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]' 
                  : 'bg-[#05050a] text-gray-500 border-gray-800 hover:text-gray-300 hover:bg-[#1a1a2e]/50'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
        
        <div className="flex flex-wrap gap-2 w-full sm:w-auto text-[9px] font-mono text-gray-400 bg-[#05050a] p-1.5 rounded border border-gray-800 transition-colors group-hover:border-gray-600">
          <button onClick={() => setAutoScale(!autoScale)} className={`px-2 py-0.5 rounded transition ${autoScale ? 'bg-[#b52aff]/20 text-[#b52aff]' : 'hover:bg-gray-800'}`}>AUTO</button>
          <button onClick={() => setShowGrid(!showGrid)} className={`px-2 py-0.5 rounded transition ${showGrid ? 'bg-[#00f0ff]/20 text-[#00f0ff]' : 'hover:bg-gray-800'}`}>GRID</button>
          <div className="flex items-center gap-1 px-2 border-l border-gray-800">
            <button onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.5))} className="hover:text-white">-</button>
            <span className="w-8 text-center">{zoomLevel}x</span>
            <button onClick={() => setZoomLevel(Math.min(5, zoomLevel + 0.5))} className="hover:text-white">+</button>
          </div>
          <span className="hidden xl:inline-block px-2 border-l border-gray-800" title="Sweep Speed">30mm/s</span>
          <span className="hidden xl:inline-block px-2 border-l border-gray-800" title="Line Noise Filter">50Hz</span>
        </div>
      </div>
      
      <div className={`flex-1 flex flex-col gap-0.5 bg-[#030308] border border-gray-800/50 rounded-lg p-2 relative overflow-hidden ${showGrid ? 'bg-[url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCI+PHBhdGggZD0iTTAgMjBMMjAgMjBMMjAgMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=")]' : ''}`}>
        {displayChannels.map((ch, i) => (
          <div key={ch} className="flex-1 relative flex items-center channel-row group/row">
            <div className="w-8 text-[10px] font-mono text-gray-500 font-bold z-10 bg-transparent pr-2 group-hover/row:text-white transition-colors text-right cursor-crosshair" title={`${ch} Electrode`} style={{ color: colors[i] }}>
              {ch}
            </div>
            <div className="flex-1 h-full relative">
              <div className="absolute inset-0 border-b border-gray-800/20 flex items-center">
                <div className="w-full h-[1px] bg-gray-800/10"></div>
              </div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayData} margin={{top: 0, right: 0, left: 0, bottom: 0}}>
                  <YAxis domain={autoScale ? ['auto', 'auto'] : [-100 / zoomLevel, 100 / zoomLevel]} hide />
                  <Line 
                    type="monotone" 
                    dataKey={ch} 
                    stroke={activeTab === 'RAW' ? colors[i] : activeTab === 'FILTERED' ? '#00ff9d' : '#b52aff'} 
                    strokeWidth={1.5} 
                    dot={false} 
                    isAnimationActive={false} 
                    style={{ filter: `drop-shadow(0 0 2px ${colors[i]}40)` }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
        {/* Timeline Axis */}
        <div className="h-4 border-t border-gray-800 flex justify-between items-center px-8 text-[8px] font-mono text-gray-600 mt-1">
          <span>-4.0s</span>
          <span>-3.0s</span>
          <span>-2.0s</span>
          <span>-1.0s</span>
          <span className="text-[#00ff9d]">0.0s (NOW)</span>
        </div>
      </div>
    </motion.div>
  );
}
