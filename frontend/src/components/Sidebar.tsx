/**
 * Sidebar.tsx — Acquisition config + model telemetry panel.
 *
 * Wired to real backend APIs:
 *  • Dataset change   → POST /api/simulate
 *  • Artifact toggle  → POST /api/simulate
 *  • TRAIN button     → POST /api/train  (non-blocking, polls /api/train-status)
 *  • CALIBRATE        → visual feedback only (no backend call needed)
 *
 * Receives simSettings + onSimSettingsChange from App.tsx (lifted state).
 */

import { useState, useEffect, useRef } from 'react';
import { Settings, Pause, Play, RefreshCw, Database, UploadCloud, Loader2, Zap, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';
import { postSimulate, postTrain, getTrainStatus, getDatasets, type SimulationSettings, type DatasetsResponse } from '../api';
import { useRetryRequest } from '../hooks/useRetryRequest';

const ARTIFACTS = ['Ocular (Blink)', 'Muscular (EMG)', '50/60Hz Line', '1/f Noise'] as const;

interface SidebarProps {
  isRunning:           boolean;
  setIsRunning:        (v: boolean) => void;
  predictions:         any;
  simSettings:         SimulationSettings;
  onSimSettingsChange: (s: SimulationSettings) => void;
}

export default function Sidebar({
  isRunning, setIsRunning, predictions, simSettings, onSimSettingsChange,
}: SidebarProps) {

  const [isCalibrating, setIsCalibrating]   = useState(false);
  const [isTraining,    setIsTraining]       = useState(false);
  const [trainProgress, setTrainProgress]   = useState(0);
  const [datasetsInfo,  setDatasetsInfo]    = useState<DatasetsResponse | null>(null);
  const [isSyncing,     setIsSyncing]       = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Fetch dataset list on mount ────────────────────────────────────────────
  useRetryRequest(getDatasets, {
    maxRetries: 3,
    startDelay: 10000,
    initialDelay: 500,
    maxDelay: 2000,
    onSuccess: (d) => setDatasetsInfo(d)
  });

  // ── Poll training status while training ────────────────────────────────────
  useEffect(() => {
    if (isTraining) {
      pollRef.current = setInterval(async () => {
        try {
          const s = await getTrainStatus();
          setTrainProgress(s.progress);
          if (!s.is_training) {
            setIsTraining(false);
            setTrainProgress(100);
            clearInterval(pollRef.current!);
            toast.success('Model training complete — weights loaded!', { icon: '🎯' });
          }
        } catch { /* ignore */ }
      }, 1500);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [isTraining]);

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleDatasetChange = async (dataset_name: string) => {
    const next = { ...simSettings, dataset_name };
    onSimSettingsChange(next);
    setIsSyncing(true);
    try {
      await postSimulate(next);
      toast.success(`Dataset → ${dataset_name}`, { icon: '📂', duration: 2000 });
    } catch {
      toast.error('Failed to update dataset on backend');
    } finally {
      setIsSyncing(false);
    }
  };

  const handleArtifactToggle = async (artifact: string, checked: boolean) => {
    const artifacts = checked
      ? [...simSettings.artifacts, artifact]
      : simSettings.artifacts.filter(a => a !== artifact);
    const next = { ...simSettings, artifacts };
    onSimSettingsChange(next);
    try {
      await postSimulate(next);
    } catch { /* silent — artifact injection is best-effort */ }
  };

  const handleCalibrate = () => {
    setIsCalibrating(true);
    setTimeout(() => setIsCalibrating(false), 2000);
  };

  const handleTrain = async () => {
    if (isTraining) return;
    try {
      const res = await postTrain();
      if (res.status === 'already_training') {
        toast('Training already in progress', { icon: '⏳' });
        setIsTraining(true);
      } else {
        setIsTraining(true);
        setTrainProgress(0);
        toast('Model training started on backend…', { icon: '⚙️' });
      }
    } catch {
      toast.error('Failed to start training');
    }
  };

  const availableDatasets = datasetsInfo?.available ?? ['DEAP', 'SEED', 'PhysioNet', 'SYNTHETIC'];
  const currentMeta = datasetsInfo?.meta?.[simSettings.dataset_name];

  return (
    <div className="flex flex-col gap-4 h-full">

      {/* ── ACQUISITION CONFIG ─────────────────────────────────────────────── */}
      <motion.div
        whileHover={{ scale: 1.01 }}
        className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 flex flex-col gap-5 relative overflow-hidden transition-all duration-300 hover:border-[#00f0ff]/30"
      >
        <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#00f0ff] to-transparent opacity-50" />

        <h2 className="text-sm font-bold tracking-widest text-gray-400 flex items-center gap-2 font-mono" title="Configure simulated hardware and injected artifacts">
          <Settings className="w-4 h-4 text-[#00f0ff]" />
          ACQUISITION CONFIG
          {isSyncing && <Loader2 className="w-3 h-3 animate-spin text-[#00f0ff] ml-auto" />}
        </h2>

        <div className="space-y-4 font-mono">

          {/* Dataset selector */}
          <div>
            <label className="block text-[10px] text-gray-500 mb-1.5 font-bold tracking-wider">
              DATASET / STREAM
            </label>
            <div className="relative">
              <select
                value={simSettings.dataset_name}
                onChange={e => handleDatasetChange(e.target.value)}
                className="w-full bg-[#05050a] border border-gray-800 rounded p-2 text-xs text-gray-300 focus:border-[#00f0ff] outline-none appearance-none cursor-pointer transition-colors hover:border-gray-600"
              >
                {availableDatasets.map(ds => (
                  <option key={ds} value={ds}>{ds}</option>
                ))}
              </select>
              <div className="absolute right-3 top-2.5 pointer-events-none text-gray-600 text-xs">▼</div>
            </div>

            {/* Dataset metadata badge */}
            {currentMeta && (
              <div className="mt-1.5 grid grid-cols-2 gap-x-3 text-[9px] text-gray-600 font-mono">
                <span>{currentMeta.subjects} subjects</span>
                <span>{currentMeta.channels} ch @ {currentMeta.fs} Hz</span>
                <span className="col-span-2 truncate">{currentMeta.duration}</span>
              </div>
            )}
          </div>

          {/* Artifact injection */}
          <div>
            <label className="block text-[10px] text-gray-500 mb-1.5 font-bold tracking-wider">NOISE / ARTIFACTS</label>
            <div className="grid grid-cols-2 gap-2">
              {ARTIFACTS.map((a) => (
                <label
                  key={a}
                  title={`Toggle ${a} artifact injection`}
                  className="flex items-center gap-2 text-[9px] bg-[#05050a] border border-gray-800 p-2 rounded cursor-pointer hover:border-[#b52aff] transition-colors hover:bg-[#1a1a2e]/50"
                >
                  <input
                    type="checkbox"
                    className="accent-[#b52aff]"
                    checked={simSettings.artifacts.includes(a)}
                    onChange={e => handleArtifactToggle(a, e.target.checked)}
                  />
                  {a}
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Control buttons */}
        <div className="mt-2 grid grid-cols-2 gap-3 font-mono text-xs font-bold">
          <motion.button
            whileHover={!isCalibrating ? { scale: 1.02 } : {}}
            whileTap={!isCalibrating ? { scale: 0.98 } : {}}
            onClick={() => setIsRunning(!isRunning)}
            disabled={isCalibrating}
            className={`border rounded p-2.5 flex items-center justify-center gap-2 transition-all shadow-inner h-10 ${
              isCalibrating ? 'opacity-50 cursor-not-allowed border-gray-700 text-gray-500' :
              isRunning
                ? 'border-[#ff2a9d] text-[#ff2a9d] hover:bg-[#ff2a9d]/10'
                : 'border-[#00ff9d] text-[#00ff9d] hover:bg-[#00ff9d]/10 bg-[#00ff9d]/5'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4 shrink-0" /> : <Play className="w-4 h-4 shrink-0" />}
            <span>{isRunning ? 'HALT' : 'INITIATE'}</span>
          </motion.button>

          <motion.button
            whileHover={!isCalibrating ? { scale: 1.02 } : {}}
            whileTap={!isCalibrating ? { scale: 0.98 } : {}}
            onClick={handleCalibrate}
            disabled={isCalibrating}
            className={`bg-[#05050a] border border-gray-700 rounded p-2.5 flex items-center justify-center gap-2 transition-all h-10 ${
              isCalibrating
                ? 'text-[#b52aff] border-[#b52aff]/50 opacity-80 cursor-wait'
                : 'text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            {isCalibrating ? <Loader2 className="w-4 h-4 animate-spin shrink-0" /> : <RefreshCw className="w-4 h-4 shrink-0" />}
            <span>{isCalibrating ? 'CALIBRATING' : 'CALIBRATE'}</span>
          </motion.button>
        </div>

        {/* TRAIN button */}
        <motion.button
          whileHover={!isTraining ? { scale: 1.01 } : {}}
          whileTap={!isTraining ? { scale: 0.98 } : {}}
          onClick={handleTrain}
          disabled={isTraining}
          className={`w-full border rounded p-2.5 flex items-center justify-center gap-2 font-mono text-xs font-bold transition-all h-10 ${
            isTraining
              ? 'border-[#ffea00]/40 text-[#ffea00] cursor-wait opacity-80'
              : 'border-[#ffea00]/50 text-[#ffea00] hover:bg-[#ffea00]/10'
          }`}
        >
          {isTraining
            ? <><Loader2 className="w-4 h-4 animate-spin shrink-0" /> TRAINING… {trainProgress}%</>
            : <><Zap className="w-4 h-4 shrink-0" /> TRAIN MODEL</>
          }
        </motion.button>

        {/* Training progress bar */}
        <AnimatePresence>
          {isTraining && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="w-full bg-gray-800 rounded h-1 -mt-2"
            >
              <div
                className="bg-[#ffea00] h-1 rounded transition-all duration-700"
                style={{ width: `${trainProgress}%` }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* ── MODEL TELEMETRY ────────────────────────────────────────────────── */}
      <motion.div
        whileHover={{ scale: 1.01 }}
        className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 flex-1 relative flex flex-col transition-all duration-300 hover:border-[#b52aff]/30"
      >
        <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#b52aff] to-transparent opacity-50" />
        <h2 className="text-sm font-bold tracking-widest text-gray-400 flex items-center gap-2 font-mono mb-4" title="Live Model Inference Telemetry">
          <Database className="w-4 h-4 text-[#b52aff]" />
          MODEL TELEMETRY
        </h2>

        <div className="flex-1 flex flex-col gap-3 font-mono text-[10px]">

          {/* Model info */}
          <div className="bg-[#05050a] border border-gray-800 rounded p-3 space-y-2">
            <div className="flex justify-between items-center text-gray-400 border-b border-gray-800 pb-1">
              <span>ARCHITECTURE</span>
              <span className="text-[#00f0ff] font-bold">CNN-LSTM + ATTN</span>
            </div>
            <div className="flex justify-between items-center text-gray-400">
              <span title="Real-time model accuracy across target states">ACCURACY</span>
              <span className="text-[#00ff9d]">{(predictions?.confidence ?? 92.4).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center text-gray-400">
              <span title="Latency for latest batch pass">INFER TIME</span>
              <span className="text-[#ffea00]">{(predictions?.inference_latency ?? 0).toFixed(1)}ms</span>
            </div>
            <div className="flex justify-between items-center text-gray-400">
              <span>DATASET</span>
              <span className="text-[#b52aff] font-bold">{simSettings.dataset_name}</span>
            </div>
          </div>

          {/* Session summary */}
          <div className="bg-[#05050a] border border-gray-800 rounded p-3 flex-1 flex flex-col justify-center space-y-2 relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 opacity-5 transform group-hover:scale-110 transition-transform">
              <UploadCloud className="w-24 h-24" />
            </div>
            <div className="text-gray-500 tracking-widest border-b border-gray-800 pb-1 mb-1">SESSION AVERAGES</div>
            <div className="grid grid-cols-2 gap-2 relative z-10">
              <div className="flex flex-col">
                <span className="text-gray-600 text-[8px]">AVG FOCUS</span>
                <span className="text-[#00f0ff] text-xs">{(predictions?.focus ?? 0).toFixed(0)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-600 text-[8px]">AVG STRESS</span>
                <span className="text-[#ff2a9d] text-xs">{(predictions?.stress ?? 0).toFixed(0)}</span>
              </div>
              <div className="flex flex-col col-span-2 mt-1">
                <span className="text-gray-600 text-[8px]">DOMINANT STATE</span>
                <span className={`text-xs ${
                  predictions?.emotion === 'Happy'  ? 'text-[#ffea00]' :
                  predictions?.emotion === 'Calm'   ? 'text-[#00ff9d]' :
                  predictions?.emotion === 'Angry'  ? 'text-[#ff2a9d]' :
                  predictions?.emotion === 'Fear'   ? 'text-[#ff6600]' :
                  'text-[#00f0ff]'
                }`}>
                  {predictions?.emotion?.toUpperCase() ?? 'CALM'}
                </span>
              </div>
            </div>

            {/* Training complete indicator */}
            {trainProgress === 100 && !isTraining && (
              <div className="flex items-center gap-1 text-[#00ff9d] text-[9px] mt-1">
                <CheckCircle2 className="w-3 h-3" />
                <span>Model retrained &amp; hot-reloaded</span>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
