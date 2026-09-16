
import { BrainCircuit } from 'lucide-react';

import { motion } from 'framer-motion';

interface StatusBarProps {
  isRunning: boolean;
  stats: any;
  latency: number;
}

export default function StatusBar({ isRunning, stats, latency }: StatusBarProps) {
  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel px-6 py-3 flex justify-between items-center border border-[#1a1a2e] rounded-xl shadow-[0_0_30px_rgba(0,0,0,0.8)] backdrop-blur-xl bg-[#0a0a1a]/90 relative overflow-hidden transition-all duration-300 hover:border-[#00f0ff]/50"
    >
      <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#00f0ff] to-transparent opacity-50"></div>
      
      <div className="flex items-center gap-4">
        <motion.div 
          animate={isRunning ? { rotate: 360 } : {}}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          className="relative"
        >
          <BrainCircuit className="text-[#00f0ff] w-8 h-8" />
          <div className="absolute inset-0 bg-[#00f0ff] blur-md opacity-40 mix-blend-screen"></div>
        </motion.div>
        <div>
          <h1 className="text-xl font-black tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-[#00f0ff] via-[#b52aff] to-[#ff2a9d]">
            NEURAL-LINK CORE
          </h1>
          <div className="text-[9px] tracking-[0.2em] text-gray-500 font-mono mt-0.5">BCI RESEARCH WORKSTATION // V3.1</div>
        </div>
      </div>
      
      <div className="flex items-center gap-6">
        {stats && (
          <div className="hidden lg:flex items-center gap-4 text-[10px] font-mono font-bold">
            <div className="flex flex-col items-end" title="CPU Usage">
              <span className="text-gray-500">CPU</span>
              <span className="text-[#00f0ff]">{stats.cpu_usage?.toFixed(1) || 0}%</span>
            </div>
            <div className="w-[1px] h-6 bg-gray-800"></div>
            <div className="flex flex-col items-end" title="GPU Usage">
              <span className="text-gray-500">GPU</span>
              <span className="text-[#b52aff]">{stats.gpu_usage?.toFixed(1) || 0}%</span>
            </div>
            <div className="w-[1px] h-6 bg-gray-800"></div>
            <div className="flex flex-col items-end" title="RAM Usage">
              <span className="text-gray-500">RAM</span>
              <span className="text-[#00ff9d]">{stats.ram_usage?.toFixed(1) || 0}%</span>
            </div>
            <div className="w-[1px] h-6 bg-gray-800"></div>
            <div className="flex flex-col items-end" title="UI Frames Per Second">
              <span className="text-gray-500">FPS</span>
              <span className="text-[#ffea00]">{stats.fps}</span>
            </div>
            <div className="w-[1px] h-6 bg-gray-800"></div>
            <div className="flex flex-col items-end" title="Model Inference Latency">
              <span className="text-gray-500">LATENCY</span>
              <span className="text-[#ff2a9d]">{latency?.toFixed(1) || '0.0'} ms</span>
            </div>
          </div>
        )}

        <div className="w-[1px] h-8 bg-gray-800 hidden lg:block"></div>

        <div className="flex items-center gap-3">
           {/* Mini Status Indicators */}
           <div className="flex gap-2 mr-2">
             <div title="WebSocket Status" className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-red-500'}`}></div>
             <div title="Database Status" className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_5px_#22c55e]"></div>
             <div title="Model Status" className={`w-1.5 h-1.5 rounded-full ${stats ? 'bg-[#00f0ff] shadow-[0_0_5px_#00f0ff]' : 'bg-gray-600'}`}></div>
           </div>
           <div className="flex items-center gap-2 bg-[#05050a] px-3 py-1.5 rounded-lg border border-gray-800 shadow-inner">
             <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-[#00ff9d] shadow-[0_0_8px_#00ff9d] animate-pulse' : 'bg-red-500'}`}></div>
             <span className="text-xs font-mono tracking-wider font-bold text-gray-300">{isRunning ? 'SYSTEM ACTIVE' : 'STANDBY'}</span>
           </div>
        </div>
      </div>
    </motion.header>
  );
}
