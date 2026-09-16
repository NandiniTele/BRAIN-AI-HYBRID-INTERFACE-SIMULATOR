
import { motion } from 'framer-motion';
import { Map } from 'lucide-react';

interface BrainMapProps {
  signals: any;
}

export default function BrainMap({ signals }: BrainMapProps) {
  // Approximate 2D coordinates for 10-20 system on a top-down circle (x, y percentages)
  const electrodes = [
    { id: 'Fp1', x: 35, y: 15 }, { id: 'Fp2', x: 65, y: 15 },
    { id: 'F7', x: 15, y: 35 }, { id: 'F3', x: 35, y: 35 }, { id: 'Fz', x: 50, y: 35 }, { id: 'F4', x: 65, y: 35 }, { id: 'F8', x: 85, y: 35 },
    { id: 'T3', x: 10, y: 50 }, { id: 'C3', x: 30, y: 50 }, { id: 'Cz', x: 50, y: 50 }, { id: 'C4', x: 70, y: 50 }, { id: 'T4', x: 90, y: 50 },
    { id: 'T5', x: 15, y: 70 }, { id: 'P3', x: 35, y: 70 }, { id: 'Pz', x: 50, y: 70 }, { id: 'P4', x: 65, y: 70 }, { id: 'T6', x: 85, y: 70 },
    { id: 'O1', x: 40, y: 90 }, { id: 'O2', x: 60, y: 90 }
  ];

  // Map signal amplitude to a color scale (blue -> red)
  const getElectrodeColor = (id: string) => {
    if (!signals || signals[id] === undefined) return '#1a1a3a';
    const val = Math.abs(signals[id]);
    // Simple heuristic: > 15 is hot (red/pink), > 5 is active (cyan), else idle (blue)
    if (val > 15) return '#ff2a9d';
    if (val > 5) return '#00f0ff';
    return '#1a1a3a';
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 relative flex flex-col items-center justify-center transition-all duration-300 hover:border-[#ff2a9d]/30 group h-full"
    >
      <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#ff2a9d] to-transparent opacity-50"></div>
      
      <div className="w-full flex justify-between items-center mb-4 absolute top-4 left-4 right-4 px-2">
        <h2 className="text-[10px] font-bold tracking-widest text-gray-400 flex items-center gap-2 font-mono" title="International 10-20 EEG Scalp Mapping">
          <Map className="w-3 h-3 text-[#ff2a9d]" />
          10-20 SCALP MAP
        </h2>
        <span className="text-[9px] font-mono text-gray-500 bg-[#05050a] px-2 py-0.5 rounded border border-gray-800 transition-colors group-hover:border-gray-600">TOP DOWN</span>
      </div>

      <div className="relative w-48 h-56 mt-8">
        {/* Head outline */}
        <div className="absolute inset-0 rounded-[50%_50%_50%_50%/40%_40%_60%_60%] border-2 border-gray-700/50 shadow-[inset_0_0_20px_rgba(0,0,0,0.8)] bg-[#05050a]"></div>
        {/* Nose */}
        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 w-6 h-6 border-t-2 border-l-2 border-gray-700/50 rotate-45 rounded-tl"></div>
        {/* Ears */}
        <div className="absolute top-1/2 -left-3 transform -translate-y-1/2 w-4 h-12 border-l-2 border-t-2 border-b-2 border-gray-700/50 rounded-l-[100%]"></div>
        <div className="absolute top-1/2 -right-3 transform -translate-y-1/2 w-4 h-12 border-r-2 border-t-2 border-b-2 border-gray-700/50 rounded-r-[100%]"></div>

        {/* Electrodes */}
        {electrodes.map((el) => {
          const color = getElectrodeColor(el.id);
          const isActive = color !== '#1a1a3a';
          const val = signals && signals[el.id] !== undefined ? signals[el.id].toFixed(2) : '0.00';
          return (
            <div 
              key={el.id}
              className="absolute transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center cursor-crosshair group/electrode"
              style={{ top: `${el.y}%`, left: `${el.x}%` }}
            >
              <div className="absolute bottom-4 opacity-0 group-hover/electrode:opacity-100 transition-opacity bg-[#05050a] text-white text-[8px] font-mono px-1.5 py-0.5 rounded border border-[#00f0ff] whitespace-nowrap z-50 pointer-events-none shadow-[0_0_10px_rgba(0,240,255,0.3)]">
                {el.id}: {val}µV
              </div>
              <motion.div 
                animate={{ backgroundColor: color, scale: isActive ? [1, 1.4, 1] : 1 }}
                transition={{ duration: 0.3, ease: "easeInOut", repeat: isActive ? Infinity : 0, repeatDelay: Math.random() * 0.5 }}
                className="w-3.5 h-3.5 rounded-full border border-gray-900 z-10"
                style={{ filter: isActive ? `drop-shadow(0 0 6px ${color})` : 'none' }}
              />
              <span className="absolute -bottom-3 text-[8px] font-mono text-gray-500 group-hover/electrode:text-white transition-colors z-20 pointer-events-none font-bold">
                {el.id}
              </span>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
