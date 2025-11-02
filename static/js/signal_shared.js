// static/js/signal_shared.js
// Shared frontend helpers for EEG/ECG pages to reduce duplication.
// Exposes a global namespace: window.SignalShared

// Immediately Invoked Function Expression (IIFE) to create isolated scope
(function(){
  // Create main namespace object for SignalShared
  const SS = {};

  // ===========================================================================
  // ENDPOINT MANAGEMENT
  // ===========================================================================

  /**
   * Build endpoint URLs from a base path
   * @param {string} base - Base path like '/eeg' or '/ecg'
   * @returns {Object} Object containing full endpoint URLs
   */
  SS.createEndpoints = function(base){
    return {
      base,                    // Base path
      config: base + '/config', // Configuration endpoint
      upload: base + '/upload', // File upload endpoint
      update: base + '/update', // Data update endpoint
      predict: base + '/predict' // Prediction endpoint
    };
  };

  // ===========================================================================
  // ECG-SPECIFIC DATA FETCHING AND DISPATCHING
  // ===========================================================================

  /**
   * Orchestrator tailored for ECG page (single time plot)
   * Handles fetching data and updating multiple visualization types
   * @param {Object} params - Configuration object
   */
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
    // Fetch data from server with POST request
    const result = await SS.fetchJson(updateUrl, { 
      method:'POST', 
      headers:{'Content-Type':'application/json'}, 
      body: JSON.stringify(payload) 
    });
    
    // Extract sampling rates from state
    const nativeFs = state.fs;
    const viewFs = state.viewFs;
    const width = state.width;
    
    // Store window size information in buffers for reference
    buffers.__windowSec = width; // Window size in seconds
    buffers.__windowPoints = Math.round(width * viewFs); // Window size in data points

    // Update time domain plot if requested
    if (display === 'all' || display === 'time'){
      // Call time plot updater and capture updated globalTime
      globalTime = SS.updateTimePlotSingle(ids.timeId, selected, result.signals||{}, nativeFs, viewFs, buffers, width, globalTime);
    }

    // Update polar plot if requested
    if (display === 'all' || display === 'polar'){
      // Update polar visualization
      SS.updatePolarPlot(ids.polarId, selected, result.signals||{}, nativeFs, viewFs, buffers, state.isCumulativePolar, SS.resampleWithAliasing, globalTime);
    }

    // Update recurrence plot if requested and we have at least 2 channels
    if ((display === 'all' || display === 'recurrence') && selected.length >= 2){
      // Extract the two channels for recurrence plot (X vs Y)
      const chX = selected[0];
      const chY = selected[1];
      // Update recurrence plot visualization
      SS.updateRecurrencePlot(ids.recurrenceId, chX, chY, result.signals||{}, nativeFs, viewFs, buffers, SS.resampleWithAliasing, names, globalTime);
    }

    // Return updated global time and result data
    return { globalTime, result };
  };

  // ===========================================================================
  // TIME DOMAIN PLOT MANAGEMENT (SINGLE PLOT WITH MULTIPLE TRACES)
  // ===========================================================================

  /**
   * Initialize a single plot with multiple traces for time domain visualization
   */
  SS.initTimePlotSingle = function(targetId, selected, nameMap, widthSec, colors, layoutOverrides){
    // Create trace for each selected channel
    const traces = selected.map((ch,i)=>({ 
      x:[], // Initialize empty x-axis data (time)
      y:[], // Initialize empty y-axis data (amplitude)
      mode:'lines', // Display as line plot
      name: nameMap[String(ch)]||`Ch ${ch+1}`, // Channel name from map or default
      line:{ color: colors[i%colors.length], width:1.5 }, // Channel color cycling
      showlegend:false, // Hide legend for cleaner display
      hoverinfo:'none' // Disable hover information for performance
    }));
    
    // Configure plot layout with dark theme
    const layout = {
      title: `Real-time Signal (time domain) - Window: ${widthSec}s`, // Plot title with window info
      paper_bgcolor:'#000', // Black background for paper
      plot_bgcolor:'#000', // Black background for plot area
      font:{ color:'#e0e0e0' }, // Light gray text color
      xaxis:{ title:'Time (s)', range:[0,widthSec], gridcolor:'#333' }, // X-axis configuration
      yaxis:{ title:'Amplitude', gridcolor:'#333', autorange:true }, // Y-axis configuration
      showlegend:false // Hide legend
    };
    
    // Apply any layout customizations provided by caller
    if (layoutOverrides && typeof layoutOverrides === 'object') {
      Object.assign(layout, layoutOverrides);
    }
    
    // Create the plot with Plotly
    Plotly.newPlot(targetId, traces, layout, { responsive:true, displayModeBar:false });
  };

  /**
   * Update a single multi-trace time plot with new data
   */
  SS.updateTimePlotSingle = function(targetId, selected, resultSignals, nativeFs, viewFs, buffers, widthSec, globalTime){
    // Calculate time step between samples at view sampling rate
    const dt = 1 / viewFs;
    
    // Determine number of samples in incoming data by finding largest chunk
    let maxIn = 0;
    // Loop through selected channels to find maximum incoming data length
    selected.forEach(ch=>{ 
      const arr = (resultSignals && resultSignals[String(ch)])||[]; 
      if (arr.length>maxIn) maxIn = arr.length; 
    });
    // Calculate number of output samples after resampling
    const M = Math.max(1, Math.round(maxIn));
    
    // Create time array for the new data chunk
    const timeChunk = new Array(M);
    for (let i=0;i<M;i++){ 
      globalTime += dt; // Increment global time by one time step
      timeChunk[i] = globalTime; // Store current time in array
    }

    // Calculate maximum number of points to keep in buffer
    const maxPoints = Math.round(widthSec * viewFs);
    
    // Process each selected channel
    selected.forEach((ch, idx)=>{
      // Get new signal data for this channel
      const chunk = (resultSignals && resultSignals[String(ch)])||[];
      
      // Initialize buffer if it doesn't exist or is malformed
      if (!buffers[ch] || !Array.isArray(buffers[ch].time)) {
        buffers[ch] = { 
          data: (buffers[ch] && Array.isArray(buffers[ch].data)) ? buffers[ch].data : [], 
          time: [] 
        };
      }
      
      // Handle resampling if incoming data length doesn't match expected output length
      let use = chunk; // Default to using raw data
      const inLen = chunk.length;
      if (inLen && inLen !== M){
        // Simple aliasing resampling: pick nearest indices to match timeChunk length
        const step = inLen / M; // Calculate step size for resampling
        const res = new Array(M); // Create output array
        for (let i=0;i<M;i++){ 
          // Map output index to input index using nearest neighbor
          res[i] = chunk[Math.min(inLen-1, Math.floor(i*step))]; 
        }
        use = res; // Use resampled data
      }
      
      // Add new data to buffer
      buffers[ch].data.push(...use);
      buffers[ch].time.push(...timeChunk);
      
      // Trim buffer if it exceeds maximum size
      if (buffers[ch].data.length > maxPoints){
        const drop = buffers[ch].data.length - maxPoints; // Calculate how many points to remove
        buffers[ch].data.splice(0, drop); // Remove oldest data points
        buffers[ch].time.splice(0, drop); // Remove corresponding time points
      }
      
      // Update the plot with new buffer data
      Plotly.restyle(targetId, { x: [buffers[ch].time], y: [buffers[ch].data] }, [idx]);
    });
    
    // Calculate new x-axis range for scrolling effect
    const xmin = globalTime - widthSec; // Start time of visible window
    const xmax = globalTime; // End time of visible window
    
    // Update plot layout with new time range and title
    Plotly.relayout(targetId, { 
      'xaxis.range': [xmin, xmax], 
      'title': `Real-time Signal (time domain) - Window: ${widthSec}s (Time: ${globalTime.toFixed(2)}s)` 
    });
    
    // Return updated global time for next iteration
    return globalTime;
  };

  // ===========================================================================
  // GENERAL DATA FETCHING AND DISPATCHING (EEG/ECG)
  // ===========================================================================

  /**
   * Main orchestrator: fetch update and dispatch to mode-specific updaters
   */
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
    // Fetch data from server with POST request
    const result = await SS.fetchJson(updateUrl, {
      method: 'POST', 
      headers: {'Content-Type':'application/json'}, 
      body: JSON.stringify(payload)
    });
    
    // Check if no file is loaded (stop condition)
    if (result.message === 'No file loaded.') {
      return { stop: true, globalTime };
    }
    
    // Validate response data based on mode
    if (payload.mode !== 'xor'){
      // For non-XOR modes, require signals and n_samples
      if (!result.signals || typeof result.n_samples === 'undefined' || result.n_samples === 0){
        // If no signals and no band power data, return early
        if (!result.band_power || Object.keys(result.band_power).length === 0) {
          return { globalTime };
        }
      }
    } else {
      // For XOR mode, check if we have either XOR data or signals
      const hasXor = Array.isArray(result.xor) || (result.xor && typeof result.xor === 'object' && Object.keys(result.xor).length>0);
      const hasSignals = result.signals && Object.keys(result.signals).length>0;
      // If no data at all, return early
      if (!hasXor && !hasSignals) {
        return { globalTime };
      }
    }

    // Extract sampling rates and calculate timing
    const fs = state.fs;
    const nativeFs = state.nativeFs;
    const width = state.width;
    const dt = 1 / fs; // Time step at view sampling rate
    const N = result.n_samples || 0; // Number of incoming samples
    
    // Calculate number of output samples after resampling
    const M = Math.max(1, Math.round((N / nativeFs) * fs));
    
    // Create time array for the new data chunk
    const timeChunk = new Array(M);
    for (let i=0;i<M;i++){ 
      globalTime += dt; // Increment global time
      timeChunk[i] = globalTime; // Store time value
    }

    // Handle different visualization modes
    if (payload.mode === 'time'){
      // TIME MODE: Update individual time plots for each channel
      const xmin = globalTime - width; // Window start time
      const xmax = globalTime; // Window end time
      
      // Process each selected channel
      selected.forEach((chIndex) => {
        const plotId = `${ids.timePrefix}${chIndex}`; // Construct plot ID
        const raw = result.signals[String(chIndex)] || []; // Get channel data
        if (!raw.length) return; // Skip if no data
        
        // Resample data if needed to match expected length
        const use = (raw.length === M) ? raw : SS.resampleWithAliasing(raw, M);
        
        // Initialize buffer if needed
        if (!buffers[chIndex] || !Array.isArray(buffers[chIndex].data) || !Array.isArray(buffers[chIndex].time)) {
          buffers[chIndex] = { data: [], time: [] };
        }
        
        // Add new data to buffer
        buffers[chIndex].data.push(...use);
        buffers[chIndex].time.push(...timeChunk);
        
        // Time-based buffer trimming: remove data outside time window
        const cutoff = globalTime - width; // Oldest time to keep
        let drop = 0; // Count of points to remove
        const tArr = buffers[chIndex].time; // Time array reference
        
        // Find first index where time is within the window
        while (drop < tArr.length && tArr[drop] < cutoff) {
          drop++;
        }
        
        // Remove old data points
        if (drop > 0){
          buffers[chIndex].data.splice(0, drop);
          buffers[chIndex].time.splice(0, drop);
        }
        
        // Prepare data for plot update
        const xUpdate = [buffers[chIndex].time];
        const yUpdate = [buffers[chIndex].data];
        
        // Update the plot
        Plotly.restyle(plotId, { x: xUpdate, y: yUpdate });
        
        // Update plot layout with new time range
        Plotly.relayout(plotId, { 
          'xaxis.range': [xmin, xmax], 
          'title': `Channel ${names[String(chIndex)]} - Time Graph - Window: ${width}s (Time: ${globalTime.toFixed(2)}s)` 
        });
      });
    } else if (payload.mode === 'polar'){
      // POLAR MODE: Update polar plot
      buffers.__windowSec = width; // Store window size in seconds
      buffers.__windowPoints = Math.round(width * fs); // Store window size in points
      SS.updatePolarPlot(ids.singleId, selected, result.signals, nativeFs, fs, buffers, state.isCumulativePolar, SS.resampleWithAliasing, globalTime);
    } else if (payload.mode === 'recurrence'){
      // RECURRENCE MODE: Update recurrence plot (requires exactly 2 channels)
      const chX = selected[0];
      const chY = selected[1];
      buffers.__windowSec = width;
      buffers.__windowPoints = Math.round(width * fs);
      SS.updateRecurrencePlot(ids.singleId, chX, chY, result.signals, nativeFs, fs, buffers, SS.resampleWithAliasing, names, globalTime);
    } else if (payload.mode === 'xor'){
      // XOR MODE: Update XOR scatter plot
      const singleId = ids.singleId;
      
      // Normalize XOR data from server (could be array or object)
      let xorArr = [];
      if (Array.isArray(result.xor)) {
        xorArr = result.xor || []; // Direct array
      } else if (result.xor && selected.length === 1) {
        xorArr = result.xor[String(selected[0])] || []; // Object with channel keys
      }
      
      // Get XOR options from state or use defaults
      const plotFs = fs; // Use current fs for time scaling
      const opts = state && state.xorOpts ? state.xorOpts : { 
        threshold: 0.05, 
        periodSec: 1.0, 
        durationSec: 10 
      };
      
      // Update XOR plot with normalized data
      SS.updateXorPlot(singleId, xorArr, plotFs, opts);
    }

    // Return updated global time
    return { globalTime };
  };

  // ===========================================================================
  // XOR (EXCLUSIVE OR) PLOT MANAGEMENT
  // ===========================================================================

  /**
   * Initialize XOR scatter plot for detecting signal changes
   */
  SS.initXorPlot = function(targetId, layoutOverrides){
    // Create single scatter trace for XOR points
    const traces = [{ 
      x: [], // Time within chunk
      y: [], // Absolute difference values
      mode: 'markers', // Scatter plot with markers
      marker: { size: 6, color: '#3b82f6' }, // Blue markers
      showlegend: false, // No legend
      hoverinfo: 'x+y' // Show both coordinates on hover
    }];
    
    // Configure plot layout with light theme
    const layout = {
      title: { text: 'XOR: Chunk N vs Chunk N-1', x: 0.5, xanchor: 'center' }, // Centered title
      paper_bgcolor: '#ffffff', // White background
      plot_bgcolor: '#ffffff', // White plot area
      font: { color: '#1e293b' }, // Dark text
      margin: { l: 60, r: 14, t: 48, b: 52 }, // Plot margins
      autosize: true, // Responsive sizing
      xaxis: { title: { text: 'Time within Chunk [s]', standoff: 10 }, gridcolor: '#e2e8f0' }, // X-axis
      yaxis: { title: { text: 'Absolute Difference', standoff: 10 }, gridcolor: '#e2e8f0' }, // Y-axis
      showlegend: false // No legend
    };
    
    // Apply layout customizations if provided
    if (layoutOverrides && typeof layoutOverrides === 'object') {
      Object.assign(layout, layoutOverrides);
    }
    
    // Create the plot
    Plotly.newPlot(targetId, traces, layout, { 
      responsive: true, 
      displayModeBar: false, 
      displaylogo: false 
    });
  };

  /**
   * Update XOR scatter plot with new difference data
   */
  SS.updateXorPlot = function(targetId, xorArray, plotFs, { threshold = 0.05, periodSec = 1.0, durationSec = 10 } = {}){
    try{
      const y = []; // Difference values above threshold
      const x = []; // Time positions within chunk
      
      // Ensure we have an array to work with
      const arr = Array.isArray(xorArray) ? xorArray : [];
      
      // Process each value in the XOR array
      for (let i = 0; i < arr.length; i++){
        const v = Math.abs(arr[i]); // Take absolute value of difference
        // Only include points above threshold
        if (v > threshold){ 
          y.push(v); // Store difference value
          // Calculate time position within the periodic chunk
          x.push((i / Math.max(1, plotFs)) % Math.max(0.001, periodSec));
        }
      }
      
      // Limit number of points to display based on duration window
      const maxPts = Math.max(1, Math.floor(durationSec * Math.max(1, plotFs)));
      const xs = x.slice(-maxPts); // Keep only recent points
      const ys = y.slice(-maxPts); // Keep only recent points
      
      // Update the plot with new data
      Plotly.restyle(targetId, { x: [xs], y: [ys], mode: ['markers'] }, [0]);
      
      // Set fixed x-axis range to show one period, auto-scale y-axis
      Plotly.relayout(targetId, { 
        'xaxis.range': [0, Math.max(0.001, periodSec)], 
        'yaxis.autorange': true 
      });
    }catch(e){
      // Silent fail - don't break application on plot errors
    }
  };

  // ===========================================================================
  // NETWORK UTILITIES
  // ===========================================================================

  /**
   * Safe JSON fetch with helpful error messages and validation
   */
  SS.fetchJson = async function(url, options={}){
    // Execute fetch request
    const resp = await fetch(url, options);
    let payload = null;
    
    // Try to parse JSON response
    try { 
      payload = await resp.json(); 
    } catch(e) {
      // Continue with null payload if JSON parsing fails
    }
    
    // Check for HTTP errors or API-level errors
    if (!resp.ok || (payload && payload.success === false)){
      // Construct meaningful error message
      const msg = (payload && (payload.message||payload.error)) || `HTTP ${resp.status}`;
      const err = new Error(msg);
      err.payload = payload; // Attach payload for debugging
      err.status = resp.status; // Attach HTTP status
      throw err; // Re-throw with enhanced error
    }
    
    // Return payload or empty object if null
    return payload || {};
  };

  // ===========================================================================
  // FILE UPLOAD AND DRAG-DROP UTILITIES
  // ===========================================================================

  /**
   * Wire up drag-and-drop functionality for file input
   */
  SS.wireDragAndDrop = function({dropZoneEl, inputEl, onFiles}){
    // Validate required parameters
    if (!dropZoneEl || !inputEl || !onFiles) return;
    
    // Handle file input changes
    inputEl.addEventListener('change', (e)=> onFiles(e.target.files));
    
    // Prevent default behaviors for all drag events
    ['dragenter','dragover','dragleave','drop'].forEach(evt => {
      dropZoneEl.addEventListener(evt, e=>{ 
        e.preventDefault(); 
        e.stopPropagation(); 
      });
    });
    
    // Visual feedback on drag over
    dropZoneEl.addEventListener('dragover', ()=> dropZoneEl.classList.add('border-accent'));
    
    // Remove visual feedback on drag leave
    dropZoneEl.addEventListener('dragleave', ()=> dropZoneEl.classList.remove('border-accent'));
    
    // Handle file drop
    dropZoneEl.addEventListener('drop', (e)=>{
      dropZoneEl.classList.remove('border-accent'); // Remove visual feedback
      onFiles(e.dataTransfer.files); // Process dropped files
    });
  };

  /**
   * Upload a single file to the server
   */
  SS.uploadSingleFile = async function({file, uploadUrl, statusEl}){
    // Create form data with single file
    const fd = new FormData();
    fd.append('file', file);
    
    // Update status if status element provided
    if (statusEl) statusEl.textContent = `Uploading: ${file.name}...`;
    
    // Execute upload request
    const payload = await SS.fetchJson(uploadUrl, { method:'POST', body: fd });
    
    // Update status on success
    if (statusEl) statusEl.textContent = `File Loaded: ${file.name}`;
    
    return payload;
  };

  /**
   * Upload multiple files to the server
   */
  SS.uploadMultipleFiles = async function({files, uploadUrl, statusEl}){
    // Create form data with multiple files
    const fd = new FormData();
    Array.from(files).forEach(f => fd.append('files', f, f.name));
    
    // Update status if status element provided
    if (statusEl) statusEl.textContent = `Uploading ${files.length} file(s)...`;
    
    // Execute upload request
    const payload = await SS.fetchJson(uploadUrl, { method:'POST', body: fd });
    
    // Update status on success
    if (statusEl) statusEl.textContent = `Upload successful`;
    
    return payload;
  };

  // ===========================================================================
  // SIGNAL PROCESSING UTILITIES
  // ===========================================================================

  /**
   * Simple aliasing decimator to fixed output length
   * Reduces input array to specified output length using nearest-neighbor sampling
   */
  SS.resampleWithAliasing = function(arr, outLen){
    const N = Array.isArray(arr) ? arr.length : 0; // Input array length
    
    // Handle edge cases with very small arrays
    if (outLen <= 1 || N <= 1) return [N>0?arr[0]:0];
    
    // Create output array with specified length
    const out = new Array(outLen);
    
    // Calculate step size for mapping input to output indices
    const step = (N - 1) / (outLen - 1);
    
    // Resample input array to output length
    for (let i=0;i<outLen;i++){
      // Calculate input index for this output position
      const idx = Math.floor(i * step);
      // Get value from input array with bounds checking
      out[i] = arr[idx] ?? arr[N-1] ?? 0; // Fallback to last element or zero
    }
    
    return out;
  };

  // ===========================================================================
  // BUFFER MANAGEMENT UTILITIES
  // ===========================================================================

  /**
   * Update buffer with new signals and time data, maintaining maximum size
   */
  SS.updateBuffer = function(bufferObj, newSignals, newTime, maxPoints){
    // Add new data to buffer
    bufferObj.data.push(...newSignals);
    
    // Add time data if provided and buffer supports it
    if (Array.isArray(bufferObj.time) && Array.isArray(newTime)){
      bufferObj.time.push(...newTime);
    }
    
    // Trim buffer if it exceeds maximum size
    if (bufferObj.data.length > maxPoints){
      const excess = bufferObj.data.length - maxPoints; // Calculate excess points
      bufferObj.data.splice(0, excess); // Remove oldest data
      if (Array.isArray(bufferObj.time)) bufferObj.time.splice(0, excess); // Remove corresponding times
    }
  };

  /**
   * Update signal buffer with optional cumulative mode
   */
  SS.updateSignalBuffer = function(arrayRef, newSignals, maxPoints, {cumulative=false}={}){
    // Add new signals to buffer
    arrayRef.push(...newSignals);
    
    // Trim buffer if not in cumulative mode and over size limit
    if (!cumulative && arrayRef.length > maxPoints){
      arrayRef.splice(0, arrayRef.length - maxPoints); // Keep only most recent data
    }
  };

  // ===========================================================================
  // UI STATE MANAGEMENT
  // ===========================================================================

  /**
   * Update button states based on application state
   */
  SS.updateButtons = function({startBtn, pauseBtn, stopBtn, canStart=true, isStreaming=false, isPaused=false, hasChannels=false}){
    // Update Start button state and appearance
    if (startBtn){
      startBtn.disabled = !canStart; // Enable/disable based on canStart flag
      
      // Update button text based on streaming state
      if (isStreaming){
        startBtn.innerHTML = isPaused 
          ? '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Continue Streaming</span>'
          : '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Running...</span>';
      } else {
        startBtn.innerHTML = '<span class="mr-1 text-lg" style="color:#00eaff;">▶</span> <span style="color:#00eaff;">Start Streaming</span>';
      }
      
      // Update visual state
      startBtn.classList.toggle('opacity-50', !canStart);
      startBtn.classList.toggle('cursor-not-allowed', !canStart);
    }
    
    // Update Pause button state
    if (pauseBtn){
      const dis = !isStreaming || isPaused; // Disable if not streaming or already paused
      pauseBtn.disabled = dis;
      pauseBtn.classList.toggle('opacity-50', dis);
      pauseBtn.classList.toggle('cursor-not-allowed', dis);
    }
    
    // Update Stop button state
    if (stopBtn){
      const dis = !isStreaming; // Disable if not streaming
      stopBtn.disabled = dis;
      stopBtn.classList.toggle('opacity-50', dis);
      stopBtn.classList.toggle('cursor-not-allowed', dis);
    }
  };

  // ===========================================================================
  // POLAR PLOT MANAGEMENT
  // ===========================================================================

  /**
   * Initialize polar plot for circular signal visualization
   */
  SS.initPolarPlot = function(targetId, selected, indexToName, widthSec, isCumulative, colors, layoutOverrides){
    // Create scatterpolar traces for each channel
    const traces = selected.map((ch, i) => ({
      r: [], // Radial data (amplitude)
      theta: [], // Angular data (phase)
      mode: 'lines', // Line plot
      line: { color: colors[i % colors.length] }, // Channel color
      name: indexToName[String(ch)] || String(ch), // Channel name
      type: 'scatterpolar' // Polar plot type
    }));
    
    // Create title based on cumulative mode
    const title = isCumulative 
      ? 'EEG - Polar Graph - CUMULATIVE MODE' 
      : `EEG - Polar Graph - Fixed Window: ${widthSec}s`;
    
    // Configure polar plot layout with dark theme
    const layout = {
      title,
      paper_bgcolor: '#000000', // Black background
      plot_bgcolor: '#000000', // Black plot area
      font: { color: '#e5e7eb', family: 'Inter' }, // Light text
      polar: { 
        bgcolor: '#000000', // Black polar background
        radialaxis: { 
          title: 'Amplitude (Normalized)', 
          color: '#e5e7eb', 
          gridcolor: '#374151' 
        }, 
        angularaxis: { 
          direction: 'clockwise', 
          rotation: 90, 
          color: '#e5e7eb', 
          gridcolor: '#374151', 
          ticksuffix: '°' 
        } 
      }
    };
    
    // Apply layout customizations
    if (layoutOverrides && typeof layoutOverrides === 'object') {
      Object.assign(layout, layoutOverrides);
    }
    
    // Create the polar plot
    Plotly.newPlot(targetId, traces, layout, { responsive: true, displayModeBar: false });
  };

  /**
   * Update polar plot with new signal data
   */
  SS.updatePolarPlot = function(targetId, selected, resultSignals, nativeFs, viewFs, buffers, isCumulative, resampleFn, globalTime){
    const rArr = []; // Array for radial data per channel
    const thetaArr = []; // Array for angular data per channel
    
    // Process each selected channel
    selected.forEach((ch) => {
      const raw = resultSignals[String(ch)] || []; // Get channel data
      if (!raw.length) { 
        // Return empty arrays if no data
        rArr.push([]); 
        thetaArr.push([]); 
        return; 
      }
      
      // Calculate output length after resampling
      const M = Math.max(1, Math.round((raw.length / nativeFs) * viewFs));
      // Resample incoming data
      const newSignals = resampleFn(raw, M);
      
      // Update channel buffer with new data
      const buf = buffers[ch].data;
      const maxPoints = Math.max(1, Math.round(viewFs)); // Default buffer size
      buf.push(...newSignals);
      
      // Trim buffer if not in cumulative mode
      if (!isCumulative){
        const limit = buffers.__windowPoints || Math.round(viewFs * (buffers.__windowSec || 5));
        if (buf.length > limit) {
          buf.splice(0, buf.length - limit); // Keep only recent data
        }
      }
      
      // Prepare polar coordinates
      const N = buf.length; // Current buffer length
      // Create angular positions (0-360 degrees)
      const theta = buf.map((_, idx) => (N > 1 ? (idx / (N - 1)) * 360 : 0));
      // Normalize amplitudes for radial axis
      const maxAbs = buf.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
      const r = buf.map(v => v / maxAbs); // Normalize to [0,1] range
      
      // Store channel data
      rArr.push(r); 
      thetaArr.push(theta);
    });
    
    // Update plot with new data
    Plotly.restyle(targetId, { r: rArr, theta: thetaArr });
    
    // Update plot title with current time
    const title = isCumulative 
      ? `EEG - Polar Graph - CUMULATIVE MODE (Time: ${globalTime.toFixed(2)}s)`
      : `EEG - Polar Graph - Fixed Window (Time: ${globalTime.toFixed(2)}s)`;
    Plotly.relayout(targetId, { title });
  };

  // ===========================================================================
  // RECURRENCE PLOT MANAGEMENT
  // ===========================================================================

  /**
   * Initialize 2D recurrence plot for phase space visualization
   */
  SS.initRecurrencePlot = function(targetId, chXName, chYName, colormap, layoutOverrides){
    // Create 2D histogram for density visualization
    const traces = [{ 
      x: [], // X-channel data
      y: [], // Y-channel data  
      type: 'histogram2d', // 2D density plot
      colorscale: colormap || 'Viridis', // Color scale
      showscale: true, // Show color scale legend
      name: 'Density' // Trace name
    }];
    
    // Configure recurrence plot layout
    const layout = {
      title: `EEG - Recurrence Graph (Phase Space Density): ${chXName} vs ${chYName}`,
      paper_bgcolor: '#000000', // Black background
      plot_bgcolor: '#000000', // Black plot area
      font: { color: '#e5e7eb', family: 'Inter' }, // Light text
      xaxis: { 
        title: `Channel ${chXName} (µV)`, 
        color: '#e5e7eb', 
        gridcolor: '#374151' 
      },
      yaxis: { 
        title: `Channel ${chYName} (µV)`, 
        color: '#e5e7eb', 
        gridcolor: '#374151' 
      }
    };
    
    // Apply layout customizations
    if (layoutOverrides && typeof layoutOverrides === 'object') {
      Object.assign(layout, layoutOverrides);
    }
    
    // Create the recurrence plot
    Plotly.newPlot(targetId, traces, layout, { responsive: true, displayModeBar: false });
  };

  /**
   * Update recurrence plot with new channel data
   */
  SS.updateRecurrencePlot = function(targetId, chXIndex, chYIndex, resultSignals, nativeFs, viewFs, buffers, resampleFn, nameMap, globalTime){
    // Get data for both channels
    const rawX = resultSignals[String(chXIndex)] || [];
    const rawY = resultSignals[String(chYIndex)] || [];
    
    // Return early if either channel has no data
    if (!rawX.length || !rawY.length) return;
    
    // Calculate output lengths for resampling
    const Mx = Math.max(1, Math.round((rawX.length / nativeFs) * viewFs));
    const My = Math.max(1, Math.round((rawY.length / nativeFs) * viewFs));
    
    // Resample both channels
    const newX = resampleFn(rawX, Mx);
    const newY = resampleFn(rawY, My);
    
    // Update buffers for both channels with window-based trimming
    [ [chXIndex, newX], [chYIndex, newY] ].forEach(([idx, arr]) => {
      const buf = buffers[idx].data;
      const limit = buffers.__windowPoints || Math.round(viewFs * (buffers.__windowSec || 5));
      buf.push(...arr);
      // Trim buffer if exceeds window size
      if (buf.length > limit) {
        buf.splice(0, buf.length - limit);
      }
    });
    
    // Get current buffer data for plotting
    const dataX = buffers[chXIndex].data;
    const dataY = buffers[chYIndex].data;
    
    // Update plot with new data
    Plotly.restyle(targetId, { x: [dataX], y: [dataY] });
    
    // Update plot title with channel names and current time
    const chX = nameMap[String(chXIndex)] || String(chXIndex);
    const chY = nameMap[String(chYIndex)] || String(chYIndex);
    Plotly.relayout(targetId, { 
      title: `EEG - Recurrence Graph (Phase Space Density): ${chX} vs ${chY} (Time: ${globalTime.toFixed(2)}s)` 
    });
  };

  // ===========================================================================
  // GLOBAL EXPORT
  // ===========================================================================

  // Expose the SignalShared namespace globally for use by other scripts
  window.SignalShared = SS;
})();