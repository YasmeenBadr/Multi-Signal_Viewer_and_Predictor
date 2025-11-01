// static/js/signal_shared.js
// Shared frontend helpers for EEG/ECG pages to reduce duplication.
// Exposes a global namespace: window.SignalShared

(function(){
  const SS = {};

  // Build endpoint set from base, e.g., '/eeg' or '/ecg'
  SS.createEndpoints = function(base){
    return {
      base,
      config: base + '/config',
      upload: base + '/upload',
      update: base + '/update',
      predict: base + '/predict'
    };
  };

  // Orchestrator tailored for ECG page (single time plot). XOR is left to page-specific logic.
  SS.fetchAndDispatchECG = async function({
    updateUrl,
    payload, // {channels, width, polar_mode, xor_threshold}
    ids,     // {timeId, polarId, recurrenceId}
    state,   // {fs, viewFs, width, isCumulativePolar}
    selected,
    buffers,
    names,
    colors,
    colormap,
    globalTime,
    display // 'time' | 'polar' | 'recurrence' | 'all'
  }){
    const result = await SS.fetchJson(updateUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const nativeFs = state.fs, viewFs = state.viewFs, width = state.width;
    buffers.__windowSec = width; buffers.__windowPoints = Math.round(width * viewFs);

    // Time plot
    if (display === 'all' || display === 'time'){
      globalTime = SS.updateTimePlotSingle(ids.timeId, selected, result.signals||{}, nativeFs, viewFs, buffers, width, globalTime);
    }

    // Polar
    if (display === 'all' || display === 'polar'){
      SS.updatePolarPlot(ids.polarId, selected, result.signals||{}, nativeFs, viewFs, buffers, state.isCumulativePolar, SS.resampleWithAliasing, globalTime);
    }

    // Recurrence (needs exactly 2 channels)
    if ((display === 'all' || display === 'recurrence') && selected.length >= 2){
      const chX = selected[0], chY = selected[1];
      SS.updateRecurrencePlot(ids.recurrenceId, chX, chY, result.signals||{}, nativeFs, viewFs, buffers, SS.resampleWithAliasing, names, globalTime);
    }

    return { globalTime, result };
  };

  // Initialize a single multi-trace time plot
  SS.initTimePlotSingle = function(targetId, selected, nameMap, widthSec, colors, layoutOverrides){
    const traces = selected.map((ch,i)=>({ x:[], y:[], mode:'lines', name: nameMap[String(ch)]||`Ch ${ch+1}`, line:{ color: colors[i%colors.length], width:1.5 }, showlegend:false, hoverinfo:'none' }));
    const layout = {
      title: `Real-time Signal (time domain) - Window: ${widthSec}s`,
      paper_bgcolor:'#000', plot_bgcolor:'#000', font:{ color:'#e0e0e0' },
      xaxis:{ title:'Time (s)', range:[0,widthSec], gridcolor:'#333' },
      yaxis:{ title:'Amplitude', gridcolor:'#333', autorange:true },
      showlegend:false
    };
    if (layoutOverrides && typeof layoutOverrides === 'object') Object.assign(layout, layoutOverrides);
    Plotly.newPlot(targetId, traces, layout, { responsive:true, displayModeBar:false });
  };

  // Update a single multi-trace time plot using per-channel buffers with {data, time}
  SS.updateTimePlotSingle = function(targetId, selected, resultSignals, nativeFs, viewFs, buffers, widthSec, globalTime){
    const dt = 1 / viewFs;
    // Determine M from largest incoming chunk (fallback if n_samples not provided)
    let maxIn = 0;
    selected.forEach(ch=>{ const arr = (resultSignals && resultSignals[String(ch)])||[]; if (arr.length>maxIn) maxIn = arr.length; });
    const M = Math.max(1, Math.round(maxIn));
    const timeChunk = new Array(M);
    for (let i=0;i<M;i++){ globalTime += dt; timeChunk[i] = globalTime; }

    const maxPoints = Math.round(widthSec * viewFs);
    selected.forEach((ch, idx)=>{
      const chunk = (resultSignals && resultSignals[String(ch)])||[];
      if (!buffers[ch] || !Array.isArray(buffers[ch].time)) buffers[ch] = { data: (buffers[ch] && Array.isArray(buffers[ch].data)) ? buffers[ch].data : [], time: [] };
      // If server chunk at nativeFs, approximate resampling by picking nearest indices to match timeChunk length
      const inLen = chunk.length;
      let use = chunk;
      if (inLen && inLen !== M){
        // simple aliasing towards M
        const step = inLen / M; const res = new Array(M);
        for (let i=0;i<M;i++){ res[i] = chunk[Math.min(inLen-1, Math.floor(i*step))]; }
        use = res;
      }
      buffers[ch].data.push(...use);
      buffers[ch].time.push(...timeChunk);
      if (buffers[ch].data.length > maxPoints){
        const drop = buffers[ch].data.length - maxPoints;
        buffers[ch].data.splice(0, drop);
        buffers[ch].time.splice(0, drop);
      }
      Plotly.restyle(targetId, { x: [buffers[ch].time], y: [buffers[ch].data] }, [idx]);
    });
    const xmin = globalTime - widthSec, xmax = globalTime;
    Plotly.relayout(targetId, { 'xaxis.range': [xmin, xmax], 'title': `Real-time Signal (time domain) - Window: ${widthSec}s (Time: ${globalTime.toFixed(2)}s)` });
    return globalTime;
  };
  

  // Orchestrator: fetch update and dispatch to mode-specific updaters
  SS.fetchAndDispatch = async function({
    updateUrl,
    payload, // {channels, width, mode}
    ids,     // {timePrefix, singleId, bandId}
    state,   // {fs, nativeFs, width, isCumulativePolar}
    selected,
    buffers,
    names,
    colors,
    bandPower, // {smaBuffer, yRangeState, bandNames, bandColors}
    globalTime
  }){
    const result = await SS.fetchJson(updateUrl, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    if (result.message === 'No file loaded.') return { stop: true, globalTime };
    // If not XOR mode, require signals/n_samples as before. For XOR, allow updates based on result.xor
    if (payload.mode !== 'xor'){
      if (!result.signals || typeof result.n_samples === 'undefined' || result.n_samples === 0){
        if (!result.band_power || Object.keys(result.band_power).length === 0) return { globalTime };
      }
    } else {
      // XOR mode: if neither xor nor signals present, nothing to do
      const hasXor = Array.isArray(result.xor) || (result.xor && typeof result.xor === 'object' && Object.keys(result.xor).length>0);
      const hasSignals = result.signals && Object.keys(result.signals).length>0;
      if (!hasXor && !hasSignals) return { globalTime };
    }

    const fs = state.fs, nativeFs = state.nativeFs, width = state.width;
    const dt = 1 / fs;
    const N = result.n_samples || 0;
    // Build time chunk with length proportional to view FS
    const M = Math.max(1, Math.round((N / nativeFs) * fs));
    const timeChunk = new Array(M);
    for (let i=0;i<M;i++){ globalTime += dt; timeChunk[i] = globalTime; }

    if (payload.mode === 'time'){
      const xmin = globalTime - width; const xmax = globalTime;
      selected.forEach((chIndex) => {
        const plotId = `${ids.timePrefix}${chIndex}`;
        const raw = result.signals[String(chIndex)] || [];
        if (!raw.length) return;
        // Ensure incoming chunk matches the view sampling length (M)
        const use = (raw.length === M) ? raw : SS.resampleWithAliasing(raw, M);
        // Defensive: make sure buffer object exists with expected shape
        if (!buffers[chIndex] || !Array.isArray(buffers[chIndex].data) || !Array.isArray(buffers[chIndex].time)) {
          buffers[chIndex] = { data: [], time: [] };
        }
        // Push new samples
        buffers[chIndex].data.push(...use);
        buffers[chIndex].time.push(...timeChunk);
        // Time-based trimming: keep only the last `width` seconds using timestamps, not sample count.
        const cutoff = globalTime - width;
        let drop = 0;
        const tArr = buffers[chIndex].time;
        // Find first index whose time is within the window
        while (drop < tArr.length && tArr[drop] < cutoff) drop++;
        if (drop > 0){
          buffers[chIndex].data.splice(0, drop);
          buffers[chIndex].time.splice(0, drop);
        }
        const xUpdate = [buffers[chIndex].time];
        const yUpdate = [buffers[chIndex].data];
        Plotly.restyle(plotId, { x: xUpdate, y: yUpdate });
        Plotly.relayout(plotId, { 'xaxis.range': [xmin, xmax], 'title': `Channel ${names[String(chIndex)]} - Time Graph - Window: ${width}s (Time: ${globalTime.toFixed(2)}s)` });
      });
    } else if (payload.mode === 'polar'){
      // Attach window hints for shared buffer trimming
      buffers.__windowSec = width; buffers.__windowPoints = Math.round(width * fs);
      SS.updatePolarPlot(ids.singleId, selected, result.signals, nativeFs, fs, buffers, state.isCumulativePolar, SS.resampleWithAliasing, globalTime);
    } else if (payload.mode === 'recurrence'){
      const chX = selected[0], chY = selected[1];
      buffers.__windowSec = width; buffers.__windowPoints = Math.round(width * fs);
      SS.updateRecurrencePlot(ids.singleId, chX, chY, result.signals, nativeFs, fs, buffers, SS.resampleWithAliasing, names, globalTime);
    } else if (payload.mode === 'xor'){
      // XOR mode expects a single-plot target and single selected channel
      const singleId = ids.singleId;
      // EEG backend returns either an array or dict; normalize
      let xorArr = [];
      if (Array.isArray(result.xor)) xorArr = result.xor || [];
      else if (result.xor && selected.length === 1) xorArr = result.xor[String(selected[0])] || [];
      // Use current fs from state as plotFs for time-within-chunk
      const plotFs = fs;
      const opts = state && state.xorOpts ? state.xorOpts : { threshold: 0.05, periodSec: 1.0, durationSec: 10 };
      SS.updateXorPlot(singleId, xorArr, plotFs, opts);
    }

    // Band power update (if present)
    

    return { globalTime };
  };

  // --- XOR shared helpers ---
  SS.initXorPlot = function(targetId, layoutOverrides){
    const traces = [{ x: [], y: [], mode: 'markers', marker: { size: 6, color: '#3b82f6' }, showlegend: false, hoverinfo: 'x+y' }];
    const layout = {
      title: { text: 'XOR: Chunk N vs Chunk N-1', x: 0.5, xanchor: 'center' },
      paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { color: '#1e293b' },
      margin: { l: 60, r: 14, t: 48, b: 52 }, autosize: true,
      xaxis: { title: { text: 'Time within Chunk [s]', standoff: 10 }, gridcolor: '#e2e8f0' },
      yaxis: { title: { text: 'Absolute Difference', standoff: 10 }, gridcolor: '#e2e8f0' },
      showlegend: false
    };
    if (layoutOverrides && typeof layoutOverrides === 'object') Object.assign(layout, layoutOverrides);
    Plotly.newPlot(targetId, traces, layout, { responsive: true, displayModeBar: false, displaylogo: false });
  };

  // Update XOR scatter points based on threshold, period and duration window
  SS.updateXorPlot = function(targetId, xorArray, plotFs, { threshold = 0.05, periodSec = 1.0, durationSec = 10 } = {}){
    try{
      const y = []; const x = [];
      const arr = Array.isArray(xorArray) ? xorArray : [];
      for (let i = 0; i < arr.length; i++){
        const v = Math.abs(arr[i]);
        if (v > threshold){ y.push(v); x.push((i / Math.max(1, plotFs)) % Math.max(0.001, periodSec)); }
      }
      const maxPts = Math.max(1, Math.floor(durationSec * Math.max(1, plotFs)));
      const xs = x.slice(-maxPts);
      const ys = y.slice(-maxPts);
      Plotly.restyle(targetId, { x: [xs], y: [ys], mode: ['markers'] }, [0]);
      Plotly.relayout(targetId, { 'xaxis.range': [0, Math.max(0.001, periodSec)], 'yaxis.autorange': true });
    }catch(e){}
  };

  // Safe JSON fetch with helpful error messages
  SS.fetchJson = async function(url, options={}){
    const resp = await fetch(url, options);
    let payload = null;
    try { payload = await resp.json(); } catch(e) {}
    if (!resp.ok || (payload && payload.success === false)){
      const msg = (payload && (payload.message||payload.error)) || `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.payload = payload;
      err.status = resp.status;
      throw err;
    }
    return payload || {};
  };

  // Drag-and-drop wiring for a drop zone and hidden file input
  SS.wireDragAndDrop = function({dropZoneEl, inputEl, onFiles}){
    if (!dropZoneEl || !inputEl || !onFiles) return;
    inputEl.addEventListener('change', (e)=> onFiles(e.target.files));
    ['dragenter','dragover','dragleave','drop'].forEach(evt => {
      dropZoneEl.addEventListener(evt, e=>{ e.preventDefault(); e.stopPropagation(); });
    });
    dropZoneEl.addEventListener('dragover', ()=> dropZoneEl.classList.add('border-accent'));
    dropZoneEl.addEventListener('dragleave', ()=> dropZoneEl.classList.remove('border-accent'));
    dropZoneEl.addEventListener('drop', (e)=>{
      dropZoneEl.classList.remove('border-accent');
      onFiles(e.dataTransfer.files);
    });
  };

  // Upload a single file with field name 'file'
  SS.uploadSingleFile = async function({file, uploadUrl, statusEl}){
    const fd = new FormData();
    fd.append('file', file);
    if (statusEl) statusEl.textContent = `Uploading: ${file.name}...`;
    const payload = await SS.fetchJson(uploadUrl, { method:'POST', body: fd });
    if (statusEl) statusEl.textContent = `File Loaded: ${file.name}`;
    return payload;
  };

  // Upload multiple files with field name 'files'
  SS.uploadMultipleFiles = async function({files, uploadUrl, statusEl}){
    const fd = new FormData();
    Array.from(files).forEach(f => fd.append('files', f, f.name));
    if (statusEl) statusEl.textContent = `Uploading ${files.length} file(s)...`;
    const payload = await SS.fetchJson(uploadUrl, { method:'POST', body: fd });
    if (statusEl) statusEl.textContent = `Upload successful`;
    return payload;
  };

  // Simple aliasing decimator to fixed length
//bnakhod output length (outLen) w N (length of arr we want to resample)
 SS.resampleWithAliasing = function(arr, outLen){
    const N = Array.isArray(arr) ? arr.length : 0;
    if (outLen <= 1 || N <= 1) return [N>0?arr[0]:0];
    //hn3ml array for output b length l output length l ahna wakhdeno input ll function
    const out = new Array(outLen);
/*step is how far to move along the input array for each output sample.
For example, if:
arr.length = 10
outLen = 5
then step = (10-1)/(5-1) = 9/4 = 2.25.
That means for each output index i, the corresponding input index is approximately i * 2.25*/
    const step = (N - 1) / (outLen - 1);
    for (let i=0;i<outLen;i++){
      const idx = Math.floor(i * step);
      out[i] = arr[idx] ?? arr[N-1] ?? 0;
    }
    /*For each output index i:
Compute the input index: i * step, then floor it (so it picks the lower integer index).
Fetch arr[idx] if it exists.
If it’s undefined, fallback to arr[N - 1] (the last element).
If that’s also undefined, fallback to 0.
So it essentially copies values from positions 0, floor(step), floor(2*step), … in the input array.
*/
    return out;
  };

  // Rolling buffer helpers
  SS.updateBuffer = function(bufferObj, newSignals, newTime, maxPoints){
    bufferObj.data.push(...newSignals);
    if (Array.isArray(bufferObj.time) && Array.isArray(newTime)){
      bufferObj.time.push(...newTime);
    }
    if (bufferObj.data.length > maxPoints){
      const excess = bufferObj.data.length - maxPoints;
      bufferObj.data.splice(0, excess);
      if (Array.isArray(bufferObj.time)) bufferObj.time.splice(0, excess);
    }
  };

  SS.updateSignalBuffer = function(arrayRef, newSignals, maxPoints, {cumulative=false}={}){
    arrayRef.push(...newSignals);
    if (!cumulative && arrayRef.length > maxPoints){
      arrayRef.splice(0, arrayRef.length - maxPoints);
    }
  };

  // Button state helper (non-op if any element missing)
  SS.updateButtons = function({startBtn, pauseBtn, stopBtn, canStart=true, isStreaming=false, isPaused=false, hasChannels=false}){
    if (startBtn){
      startBtn.disabled = !canStart;
      if (isStreaming){
        startBtn.innerHTML = isPaused ? '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Continue Streaming</span>'
                                      : '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Running...</span>';
      } else {
        startBtn.innerHTML = '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Start Streaming</span>';
      }
      startBtn.classList.toggle('opacity-50', !canStart);
      startBtn.classList.toggle('cursor-not-allowed', !canStart);
    }
    if (pauseBtn){
      const dis = !isStreaming || isPaused;
      pauseBtn.disabled = dis;
      pauseBtn.classList.toggle('opacity-50', dis);
      pauseBtn.classList.toggle('cursor-not-allowed', dis);
    }
    if (stopBtn){
      const dis = !isStreaming;
      stopBtn.disabled = dis;
      stopBtn.classList.toggle('opacity-50', dis);
      stopBtn.classList.toggle('cursor-not-allowed', dis);
    }
  };

  
  // --- Polar/Recurrence shared helpers ---
  SS.initPolarPlot = function(targetId, selected, indexToName, widthSec, isCumulative, colors, layoutOverrides){
    const traces = selected.map((ch, i) => ({
      r: [], theta: [], mode: 'lines', line: { color: colors[i % colors.length] }, name: indexToName[String(ch)] || String(ch), type: 'scatterpolar'
    }));
    const title = isCumulative ? 'EEG - Polar Graph - CUMULATIVE MODE' : `EEG - Polar Graph - Fixed Window: ${widthSec}s`;
    const layout = {
      title,
      paper_bgcolor: '#000000', plot_bgcolor: '#000000', font: { color: '#e5e7eb', family: 'Inter' },
      polar: { bgcolor: '#000000', radialaxis: { title: 'Amplitude (Normalized)', color: '#e5e7eb', gridcolor: '#374151' }, angularaxis: { direction: 'clockwise', rotation: 90, color: '#e5e7eb', gridcolor: '#374151', ticksuffix: '°' } }
    };
    if (layoutOverrides && typeof layoutOverrides === 'object') Object.assign(layout, layoutOverrides);
    Plotly.newPlot(targetId, traces, layout, { responsive: true, displayModeBar: false });
  };

  SS.updatePolarPlot = function(targetId, selected, resultSignals, nativeFs, viewFs, buffers, isCumulative, resampleFn, globalTime){
    const rArr = [], thetaArr = [];
    selected.forEach((ch) => {
      const raw = resultSignals[String(ch)] || [];
      if (!raw.length) { rArr.push([]); thetaArr.push([]); return; }
      const M = Math.max(1, Math.round((raw.length / nativeFs) * viewFs));
      const newSignals = resampleFn(raw, M);
      // update buffer
      const buf = buffers[ch].data; const maxPoints = Math.max(1, Math.round(viewFs)); // width handled by caller when trimming
      buf.push(...newSignals);
      if (!isCumulative){
        const limit = buffers.__windowPoints || Math.round(viewFs * (buffers.__windowSec || 5));
        if (buf.length > limit) buf.splice(0, buf.length - limit);
      }
      const N = buf.length;
      const theta = buf.map((_, idx) => (N > 1 ? (idx / (N - 1)) * 360 : 0));
      const maxAbs = buf.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
      const r = buf.map(v => v / maxAbs);
      rArr.push(r); thetaArr.push(theta);
    });
    Plotly.restyle(targetId, { r: rArr, theta: thetaArr });
    const title = isCumulative ? `EEG - Polar Graph - CUMULATIVE MODE (Time: ${globalTime.toFixed(2)}s)` : `EEG - Polar Graph - Fixed Window (Time: ${globalTime.toFixed(2)}s)`;
    Plotly.relayout(targetId, { title });
  };

  SS.initRecurrencePlot = function(targetId, chXName, chYName, colormap, layoutOverrides){
    const traces = [{ x: [], y: [], type: 'histogram2d', colorscale: colormap || 'Viridis', showscale: true, name: 'Density' }];
    const layout = {
      title: `EEG - Recurrence Graph (Phase Space Density): ${chXName} vs ${chYName}`,
      paper_bgcolor: '#000000', plot_bgcolor: '#000000', font: { color: '#e5e7eb', family: 'Inter' },
      xaxis: { title: `Channel ${chXName} (µV)`, color: '#e5e7eb', gridcolor: '#374151' },
      yaxis: { title: `Channel ${chYName} (µV)`, color: '#e5e7eb', gridcolor: '#374151' }
    };
    if (layoutOverrides && typeof layoutOverrides === 'object') Object.assign(layout, layoutOverrides);
    Plotly.newPlot(targetId, traces, layout, { responsive: true, displayModeBar: false });
  };

  SS.updateRecurrencePlot = function(targetId, chXIndex, chYIndex, resultSignals, nativeFs, viewFs, buffers, resampleFn, nameMap, globalTime){
    const rawX = resultSignals[String(chXIndex)] || [];
    const rawY = resultSignals[String(chYIndex)] || [];
    if (!rawX.length || !rawY.length) return;
    const Mx = Math.max(1, Math.round((rawX.length / nativeFs) * viewFs));
    const My = Math.max(1, Math.round((rawY.length / nativeFs) * viewFs));
    const newX = resampleFn(rawX, Mx);
    const newY = resampleFn(rawY, My);
    // update buffers (fixed window derived from buffers.__windowSec if present)
    [ [chXIndex, newX], [chYIndex, newY] ].forEach(([idx, arr]) => {
      const buf = buffers[idx].data; const limit = buffers.__windowPoints || Math.round(viewFs * (buffers.__windowSec || 5));
      buf.push(...arr);
      if (buf.length > limit) buf.splice(0, buf.length - limit);
    });
    const dataX = buffers[chXIndex].data;
    const dataY = buffers[chYIndex].data;
    Plotly.restyle(targetId, { x: [dataX], y: [dataY] });
    const chX = nameMap[String(chXIndex)] || String(chXIndex);
    const chY = nameMap[String(chYIndex)] || String(chYIndex);
    Plotly.relayout(targetId, { title: `EEG - Recurrence Graph (Phase Space Density): ${chX} vs ${chY} (Time: ${globalTime.toFixed(2)}s)` });
  };

  window.SignalShared = SS;
})();
