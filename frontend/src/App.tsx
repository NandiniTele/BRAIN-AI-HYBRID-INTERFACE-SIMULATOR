import { useState, useEffect, useRef, useCallback, lazy, Suspense } from 'react';
import StatusBar from './components/StatusBar';
import Sidebar from './components/Sidebar';
import { toast, Toaster } from 'react-hot-toast';
import { getHealth, postSimulate, WS_URL, type SimulationSettings } from './api';
import { useRetryRequest } from './hooks/useRetryRequest';

const Oscilloscope      = lazy(() => import('./components/Oscilloscope'));
const BrainMap          = lazy(() => import('./components/BrainMap'));
const CognitiveDashboard = lazy(() => import('./components/CognitiveDashboard'));
const TelemetryTable    = lazy(() => import('./components/TelemetryTable'));
const ResearchDashboard = lazy(() => import('./components/ResearchDashboard'));
const InferenceEngine   = lazy(() => import('./components/InferenceEngine'));

// ── Connection status types ───────────────────────────────────────────────────
type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

const FALLBACK = <div className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 h-full flex items-center justify-center text-[#00f0ff] font-mono text-xs animate-pulse">LOADING...</div>;

export default function App() {
  // ── EEG / inference state ─────────────────────────────────────────────────
  const [eegData,          setEegData]          = useState<any[]>([]);
  const [predictions,      setPredictions]      = useState<any>(null);
  const [bands,            setBands]            = useState<any>(null);
  const [systemInfo,       setSystemInfo]       = useState<any>(null);
  const [currentSignals,   setCurrentSignals]   = useState<any>(null);
  const [history,          setHistory]          = useState<any[]>([]);

  // ── Research mode state ───────────────────────────────────────────────────
  const [isResearchMode,   setIsResearchMode]   = useState(false);
  const [xai,              setXai]              = useState<any>(null);
  const [signalQuality,    setSignalQuality]    = useState<any>(null);
  const [researchData,     setResearchData]     = useState<any>(null);
  const [training,         setTraining]         = useState<any>(null);
  const [modelStats,       setModelStats]       = useState<any>(null);
  const [modelComparison,  setModelComparison]  = useState<any>(null);
  const [datasetMeta,      setDatasetMeta]      = useState<any>(null);
  const [sessionAnalytics, setSessionAnalytics] = useState<any>(null);
  const [regionActivity,   setRegionActivity]   = useState<any>(null);
  const [channelAnalytics, setChannelAnalytics] = useState<any>(null);
  const [aiAssistant,      setAiAssistant]      = useState<string>('');

  // ── Control / simulation settings ─────────────────────────────────────────
  const [isRunning,        setIsRunning]        = useState(true);
  const [connStatus,       setConnStatus]       = useState<ConnStatus>('connecting');
  const [simSettings,      setSimSettings]      = useState<SimulationSettings>({
    dataset_name: 'PhysioNet',
    artifacts: ['Ocular (Blink)', '1/f Noise'],
  });

  const wsRef           = useRef<WebSocket | null>(null);
  const reconnectTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay  = useRef(1000);
  const isConnecting    = useRef(false); // guard against double-open

  const { loading: isSyncingBackend } = useRetryRequest(getHealth, {
    maxRetries: 6, // Try for ~15 seconds total
    startDelay: 10000,
    initialDelay: 500,
    maxDelay: 4000,
    onSuccess: (h) => {
      toast.success(`Backend connected — v${h.version} | DB: ${h.db_status}`, { icon: '🔌' });
      // Sync initial dataset from backend and send our default artifacts
      const initialSettings = { dataset_name: h.current_dataset, artifacts: ['Ocular (Blink)', '1/f Noise'] };
      setSimSettings(initialSettings);
      postSimulate(initialSettings).catch(() => {});
    },
    onError: () => {
      toast.error('Backend unreachable — running in offline mode', { icon: '⚠️' });
      setConnStatus('error');
    }
  });

  // ── WebSocket connection with auto-reconnect ──────────────────────────────
  // Keep a stable ref to isRunning so closures in WS callbacks see current value
  const isRunningRef = useRef(isRunning);
  useEffect(() => { isRunningRef.current = isRunning; }, [isRunning]);

  const connectWS = useCallback(() => {
    if (!isRunningRef.current) return;
    // Prevent duplicate sockets: bail if one is already open OR being opened
    if (isConnecting.current) return;
    if (wsRef.current &&
        (wsRef.current.readyState === WebSocket.OPEN ||
         wsRef.current.readyState === WebSocket.CONNECTING)) return;

    isConnecting.current = true;
    setConnStatus('connecting');
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      isConnecting.current = false;
      setConnStatus('connected');
      reconnectDelay.current = 1000; // reset back-off
      toast.success('Neural stream connected', { id: 'ws-conn', icon: '🧠' });
      
      // Keep-alive ping every 20s to prevent proxy/browser idle disconnects
      const pingInterval = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send('ping');
        }
      }, 20000);
      ws.addEventListener('close', () => clearInterval(pingInterval));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      setEegData(prev => [...prev, { time: data.timestamp, ...data.signals }].slice(-150));
      setCurrentSignals(data.signals);
      setPredictions(data.predictions);
      setBands(data.bands);
      setSystemInfo(data.system);

      setXai(data.xai);
      setSignalQuality(data.signal_quality);
      setResearchData(data.research_data);
      setTraining(data.training);
      setModelStats(data.model_stats);
      setModelComparison(data.model_comparison);
      setDatasetMeta(data.dataset_meta);
      setSessionAnalytics(data.session);
      setRegionActivity(data.region_activity);
      setChannelAnalytics(data.channel_analytics);
      setAiAssistant(data.ai_assistant);

      setHistory(prev => [...prev, {
        time:       data.timestamp,
        focus:      data.predictions.focus,
        attention:  data.predictions.attention,
        stress:     data.predictions.stress,
        confidence: data.predictions.confidence,
      }].slice(-60));
    };

    ws.onerror = () => {
      isConnecting.current = false;
      setConnStatus('error');
    };

    ws.onclose = (event) => {
      isConnecting.current = false;
      if (wsRef.current === ws) {
        setConnStatus('disconnected');
        wsRef.current = null;
      }

      // Do not reconnect if closed intentionally
      if (event.code === 1000) return;

      if (isRunningRef.current) {
        // Exponential back-off reconnect (capped at 10s)
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
        reconnectTimer.current = setTimeout(() => {
          reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, 10000);
          connectWS();
        }, reconnectDelay.current);
      }
    };
  }, []); // stable — reads live values via refs

  useEffect(() => {
    // Only connect once the backend health-check succeeds
    if (isRunning && !isSyncingBackend) {
      connectWS();
    } else {
      // Tear down existing socket when paused / backend gone
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close(1000, 'Intentional close');
        wsRef.current = null;
      }
      isConnecting.current = false;
      if (!isSyncingBackend) setConnStatus('disconnected');
    }

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
        wsRef.current = null;
      }
      isConnecting.current = false;
    };
  // connectWS is intentionally omitted — it's stable (no deps) and reading
  // isRunning via isRunningRef avoids the double-fire on identity change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning, isSyncingBackend]);

  if (isSyncingBackend) {
    return (
      <div className="min-h-screen bg-[#020205] flex flex-col items-center justify-center text-[#e0e0e0] font-sans relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gray-900">
           <div className="h-1 bg-[#00f0ff] animate-pulse"></div>
        </div>
        <div className="glass-panel p-8 rounded-2xl border border-[#1a1a2e] bg-[#0a0a1a]/90 flex flex-col items-center gap-6 shadow-[0_0_50px_rgba(0,240,255,0.05)]">
           <div className="relative">
             <div className="w-16 h-16 border-4 border-gray-800 border-t-[#00f0ff] rounded-full animate-spin"></div>
             <div className="w-8 h-8 border-4 border-gray-800 border-b-[#b52aff] rounded-full animate-[spin_1.5s_linear_reverse_infinite] absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"></div>
           </div>
           <div className="text-center">
             <h2 className="text-[#00f0ff] font-mono font-bold tracking-widest mb-2">NEURAL-LINK CORE</h2>
             <p className="text-gray-400 text-xs font-mono animate-pulse">Synchronizing with FastAPI backend...</p>
           </div>
        </div>
      </div>
    );
  }

  // ── Connection status badge ───────────────────────────────────────────────
  const statusColor: Record<ConnStatus, string> = {
    connected:    'text-[#00ff9d]',
    connecting:   'text-[#ffea00] animate-pulse',
    disconnected: 'text-gray-500',
    error:        'text-[#ff2a9d]',
  };
  const statusLabel: Record<ConnStatus, string> = {
    connected:    '● LIVE',
    connecting:   '◌ CONNECTING',
    disconnected: '○ HALTED',
    error:        '✗ ERROR',
  };

  return (
    <div className="min-h-screen bg-[#020205] text-[#e0e0e0] font-sans overflow-x-hidden flex flex-col p-4 gap-4"
         style={{
           backgroundImage: 'radial-gradient(circle at 50% 0%, rgba(138, 43, 226, 0.05), transparent 40%), radial-gradient(circle at 100% 100%, rgba(0, 240, 255, 0.03), transparent 30%)'
         }}>

      <Toaster position="top-right" toastOptions={{ style: { background: '#0a0a1a', color: '#fff', border: '1px solid #333' } }} />

      {/* HEADER */}
      <StatusBar isRunning={isRunning} stats={systemInfo} latency={predictions?.inference_latency} />

      {/* TOP CONTROL BAR */}
      <div className="flex justify-between items-center pr-2 -mt-2">
        {/* Connection status pill */}
        <span className={`text-[10px] font-mono font-bold tracking-widest px-3 py-1 rounded-full border border-gray-800 bg-[#05050a] ${statusColor[connStatus]}`}>
          {statusLabel[connStatus]}
        </span>

        {/* Research mode toggle */}
        <label className="flex items-center cursor-pointer gap-2 bg-[#05050a] px-3 py-1 rounded-full border border-gray-800 hover:border-[#b52aff] transition-colors">
          <div className="relative">
            <input type="checkbox" className="sr-only" checked={isResearchMode} onChange={() => {
              setIsResearchMode(!isResearchMode);
              toast.success(isResearchMode ? 'Normal Mode Activated' : 'Research Mode Activated', { icon: isResearchMode ? '🔵' : '🔬' });
            }} />
            <div className="block bg-gray-700 w-8 h-5 rounded-full"></div>
            <div className={`dot absolute left-1 top-1 bg-white w-3 h-3 rounded-full transition ${isResearchMode ? 'transform translate-x-3 bg-[#b52aff]' : ''}`}></div>
          </div>
          <span className="text-[10px] font-mono font-bold tracking-widest text-gray-400">RESEARCH MODE</span>
        </label>
      </div>

      {/* MAIN CONTENT GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">

        {/* LEFT COLUMN — Settings */}
        <div className="lg:col-span-3 xl:col-span-2">
          <Sidebar
            isRunning={isRunning}
            setIsRunning={setIsRunning}
            predictions={predictions}
            simSettings={simSettings}
            onSimSettingsChange={setSimSettings}
          />
        </div>

        {/* CENTER COLUMN — Signal Viz */}
        <div className="lg:col-span-6 xl:col-span-6 flex flex-col gap-4">
          <div className="flex-1 min-h-[400px]">
            <Suspense fallback={FALLBACK}>
              <Oscilloscope eegData={eegData} />
            </Suspense>
          </div>
          <div className="h-64 hidden xl:block">
            <Suspense fallback={FALLBACK}>
              <TelemetryTable />
            </Suspense>
          </div>
        </div>

        {/* RIGHT COLUMN — AI & Brain Map */}
        <div className="lg:col-span-3 xl:col-span-4 flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="col-span-2 xl:col-span-1">
              <Suspense fallback={FALLBACK}>
                <BrainMap signals={currentSignals} />
              </Suspense>
            </div>
            <div className="col-span-2 xl:col-span-1">
              <Suspense fallback={FALLBACK}>
                <InferenceEngine predictions={predictions} />
              </Suspense>
            </div>
          </div>
          <div className="flex-1">
            <Suspense fallback={FALLBACK}>
              <CognitiveDashboard predictions={predictions} bands={bands} history={history} />
            </Suspense>
          </div>
        </div>
      </div>

      {/* RESEARCH MODE DASHBOARD */}
      {isResearchMode && (
        <Suspense fallback={FALLBACK}>
          <ResearchDashboard
            xai={xai}
            signalQuality={signalQuality}
            researchData={researchData}
            training={training}
            modelStats={modelStats}
            modelComparison={modelComparison}
            datasetMeta={datasetMeta}
            sessionAnalytics={sessionAnalytics}
            system={systemInfo}
            regionActivity={regionActivity}
            channelAnalytics={channelAnalytics}
            aiAssistant={aiAssistant}
          />
        </Suspense>
      )}

      {/* Mobile Telemetry */}
      <div className="h-64 block xl:hidden mt-4">
        <Suspense fallback={FALLBACK}>
          <TelemetryTable />
        </Suspense>
      </div>
    </div>
  );
}
