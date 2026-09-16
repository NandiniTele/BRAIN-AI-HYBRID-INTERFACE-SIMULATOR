import { motion } from 'framer-motion';

export default function InferenceEngine({ predictions }: { predictions: any }) {
  const data = [
    { label: 'EMOTION', value: predictions?.emotion || 'Neutral', isText: true, color: '#b52aff' },
    { label: 'ATTENTION', value: predictions?.attention || 0, color: '#00ff9d' },
    { label: 'FOCUS', value: predictions?.focus || 0, color: '#00f0ff' },
    { label: 'STRESS', value: predictions?.stress || 0, color: '#ff2a9d' },
    { label: 'FATIGUE', value: predictions?.fatigue || 0, color: '#ffea00' },
    { label: 'CONFIDENCE', value: predictions?.confidence || 0, color: '#ffffff' }
  ];

  return (
    <div className="glass-panel p-4 rounded-xl border border-[#1a1a2e] h-full flex flex-col justify-between transition-all duration-300 hover:border-[#b52aff]/30">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-[10px] font-mono text-gray-400 tracking-widest font-bold">INFERENCE ENGINE</h3>
        <div className="flex gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00ff9d] animate-pulse"></span>
          <span className="w-1.5 h-1.5 rounded-full bg-[#00ff9d] animate-pulse delay-75"></span>
          <span className="w-1.5 h-1.5 rounded-full bg-[#00ff9d] animate-pulse delay-150"></span>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        {data.map((item, idx) => (
          <div key={idx} className="flex flex-col gap-1">
            <div className="flex justify-between text-[9px] font-mono">
              <span className="text-gray-500">{item.label}</span>
              <span style={{ color: item.color }}>
                {item.isText ? item.value : `${Math.round(item.value)}%`}
              </span>
            </div>
            {!item.isText && (
              <div className="w-full h-1.5 bg-[#05050a] rounded-full overflow-hidden border border-gray-800">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: item.color, boxShadow: `0 0 8px ${item.color}80` }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(Math.max(item.value, 0), 100)}%` }}
                  transition={{ type: "spring", stiffness: 100, damping: 20 }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
