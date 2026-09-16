import { useState, useEffect } from 'react';
import { Database, Download, Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { getLogs } from '../api';

export default function TelemetryTable() {
  const [logs, setLogs] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(0);
  const [sortConfig, setSortConfig] = useState<{key: string, direction: 'asc'|'desc'} | null>(null);
  const [filterState, setFilterState] = useState('ALL');
  const rowsPerPage = 10;

  // Delay the first fetch so we don't spam ECONNREFUSED errors during the
  // ~7-10 second window while the Python backend is still starting up.
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    const fetchLogs = async () => {
      try {
        const data = await getLogs(500);
        setLogs(data);
      } catch {
        // silently ignore — backend may still be starting
      }
    };

    // Wait 10 s before the first attempt, then poll every 5 s
    const initialTimer = setTimeout(() => {
      fetchLogs();
      interval = setInterval(fetchLogs, 5000);
    }, 10_000);

    return () => {
      clearTimeout(initialTimer);
      if (interval) clearInterval(interval);
    };
  }, []);

  let filteredLogs = logs.filter(log => {
    const matchesSearch = log.session_id?.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          log.emotion?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          log.id?.toString().includes(searchTerm);
    const matchesFilter = filterState === 'ALL' || log.emotion?.toUpperCase() === filterState.toUpperCase();
    return matchesSearch && matchesFilter;
  });

  if (sortConfig !== null) {
    filteredLogs.sort((a, b) => {
      let valA = a[sortConfig.key];
      let valB = b[sortConfig.key];
      if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
      if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }

  const paginatedLogs = filteredLogs.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const handleExportCSV = () => {
    if (logs.length === 0) return;
    const headers = Object.keys(logs[0]).join(',');
    const rows = logs.map(log => Object.values(log).join(',')).join('\n');
    const blob = new Blob([`${headers}\n${rows}`], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural_telemetry_${new Date().getTime()}.csv`;
    a.click();
  };

  const handleExportJSON = () => {
    if (logs.length === 0) return;
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural_telemetry_${new Date().getTime()}.json`;
    a.click();
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/80 h-full flex flex-col relative overflow-hidden transition-all duration-300 hover:border-gray-700"
    >
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-[10px] font-bold tracking-widest text-gray-400 flex items-center gap-2 font-mono" title="Live SQLite Database Mirror">
          <Database className="w-3 h-3 text-gray-500" />
          SQLITE TELEMETRY LOGS
        </h2>
        <div className="flex gap-2">
          <div className="relative flex items-center" title="Filter by Session ID or Emotion">
            <Search className="w-3 h-3 text-gray-500 absolute left-2" />
            <input 
              type="text" 
              placeholder="SEARCH..." 
              value={searchTerm}
              onChange={(e) => {setSearchTerm(e.target.value); setPage(0);}}
              className="bg-[#05050a] border border-gray-800 rounded pl-7 pr-2 py-1 text-[9px] font-mono text-gray-300 focus:border-[#00f0ff] outline-none transition-colors"
            />
          </div>
          <div className="relative flex items-center" title="Filter by Emotion">
            <Filter className="w-3 h-3 text-gray-500 absolute left-2" />
            <select 
              value={filterState}
              onChange={(e) => {setFilterState(e.target.value); setPage(0);}}
              className="bg-[#05050a] border border-gray-800 rounded pl-7 pr-2 py-1 text-[9px] font-mono text-gray-300 focus:border-[#00f0ff] outline-none transition-colors appearance-none"
            >
              <option value="ALL">ALL STATES</option>
              <option value="CALM">CALM</option>
              <option value="HAPPY">HAPPY</option>
              <option value="SAD">SAD</option>
              <option value="ANGRY">ANGRY</option>
              <option value="FEAR">FEAR</option>
            </select>
          </div>
          <button onClick={handleExportCSV} className="text-[9px] font-mono flex items-center gap-1 bg-[#05050a] border border-gray-800 px-2 py-1 rounded text-[#00f0ff] hover:bg-[#00f0ff]/10 transition-colors" title="Export entire DB to CSV">
            <Download className="w-3 h-3" /> CSV
          </button>
          <button onClick={handleExportJSON} className="text-[9px] font-mono flex items-center gap-1 bg-[#05050a] border border-gray-800 px-2 py-1 rounded text-[#b52aff] hover:bg-[#b52aff]/10 transition-colors" title="Export entire DB to JSON">
            <Download className="w-3 h-3" /> JSON
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-auto border border-gray-800 rounded bg-[#030308] relative">
        <table className="w-full text-left text-[9px] font-mono text-gray-400">
          <thead className="bg-[#111122] text-gray-500 sticky top-0 z-10 shadow-md">
            <tr>
              <th className="p-2 pl-4 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('id')}>ID {sortConfig?.key === 'id' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('session_id')}>SESSION {sortConfig?.key === 'session_id' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('timestamp')}>TIME {sortConfig?.key === 'timestamp' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('focus')}>FOCUS {sortConfig?.key === 'focus' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('attention')}>ATTN {sortConfig?.key === 'attention' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('stress')}>STRESS {sortConfig?.key === 'stress' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('emotion')}>STATE {sortConfig?.key === 'emotion' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
              <th className="p-2 cursor-pointer hover:text-white transition-colors" onClick={() => requestSort('inference_latency')}>LATENCY {sortConfig?.key === 'inference_latency' ? (sortConfig.direction === 'asc' ? '↑' : '↓') : ''}</th>
            </tr>
          </thead>
          <tbody>
            {paginatedLogs.map((log) => (
              <tr key={log.id} className="border-t border-gray-800/50 hover:bg-[#1a1a3a] transition-colors cursor-pointer" title={`Detailed view for record #${log.id}`}>
                <td className="p-2 pl-4 text-gray-600">#{log.id}</td>
                <td className="p-2 text-gray-500">{log.session_id}</td>
                <td className="p-2">{new Date(log.timestamp * 1000).toLocaleTimeString()}</td>
                <td className="p-2 text-[#00ff9d]">{log.focus?.toFixed(1)}</td>
                <td className="p-2 text-[#b52aff]">{log.attention?.toFixed(1)}</td>
                <td className="p-2 text-[#ff2a9d]">{log.stress?.toFixed(1)}</td>
                <td className="p-2">
                  <span className={`px-1.5 py-0.5 rounded border ${
                    log.emotion === 'Calm'  ? 'text-[#00f0ff] border-[#00f0ff]/30 bg-[#00f0ff]/10' :
                    log.emotion === 'Happy' ? 'text-[#ffea00] border-[#ffea00]/30 bg-[#ffea00]/10' :
                    log.emotion === 'Sad'   ? 'text-[#b52aff] border-[#b52aff]/30 bg-[#b52aff]/10' :
                    log.emotion === 'Angry' ? 'text-[#ff2a9d] border-[#ff2a9d]/30 bg-[#ff2a9d]/10' :
                    log.emotion === 'Fear'  ? 'text-[#ff8c00] border-[#ff8c00]/30 bg-[#ff8c00]/10' :
                    'text-[#00ff9d] border-[#00ff9d]/30 bg-[#00ff9d]/10'
                  }`}>
                    {log.emotion?.toUpperCase()}
                  </span>
                </td>
                <td className="p-2 text-gray-500">{log.inference_latency?.toFixed(1)}ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex justify-between items-center text-[8px] font-mono text-gray-500">
        <span>SHOWING {Math.min(filteredLogs.length, rowsPerPage)} OF {filteredLogs.length} RECORDS</span>
        <div className="flex gap-1 items-center">
          <button 
            disabled={page === 0} 
            onClick={() => setPage(p => p - 1)}
            className="p-1 bg-gray-800 rounded disabled:opacity-50 hover:bg-gray-700 transition-colors"
            title="Previous Page"
          >
            <ChevronLeft className="w-3 h-3" />
          </button>
          <span className="px-2">PG {page + 1}</span>
          <button 
            disabled={(page + 1) * rowsPerPage >= filteredLogs.length}
            onClick={() => setPage(p => p + 1)}
            className="p-1 bg-gray-800 rounded disabled:opacity-50 hover:bg-gray-700 transition-colors"
            title="Next Page"
          >
            <ChevronRight className="w-3 h-3" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
