/* Shared front-end utilities for EEG/ECG pages */
(function(global){
  function clamp(n, min, max){ return Math.max(min, Math.min(max, n)); }
  function throttle(fn, wait){ let t=0; return function(){ const now=Date.now(); if(now-t>=wait){ t=now; return fn.apply(this, arguments);} }; }

  function createSamplingController(cfg){
    const state = {
      nativeFs: cfg.nativeFs || null,
      maxFsCap: cfg.maxFsCap || 500,
      currentFs: null,
      backend: cfg.backend || null
    };
    const els = {
      slider: cfg.sliderEl,
      value: cfg.valueEl,
      applyBtn: cfg.applyBtn || null,
      resetBtn: cfg.resetBtn || null,
      downsample: cfg.downsampleEl || null,
      downsampleValue: cfg.downsampleValueEl || null,
      aliasContainer: cfg.aliasContainer || null
    };

    function renderFsLabel(val){ if(els.value){ els.value.innerText = String(val) + ' Hz'; } }

    function syncDownsampleFromFs(){
      if(!els.downsample || !state.nativeFs || !state.currentFs) return;
      const factor = Math.max(1, Math.round(state.nativeFs / state.currentFs));
      els.downsample.value = String(factor);
      if(els.downsampleValue) els.downsampleValue.textContent = factor + 'x';
      if(cfg.onAliasingLevel){
        let level = 'none';
        if(factor >= 16) level = 'high'; else if(factor >= 8) level = 'med'; else if(factor > 1) level = 'low';
        cfg.onAliasingLevel(level);
      }
    }

    function syncFsFromDownsample(){
      if(!els.downsample || !state.nativeFs) return;
      const factor = Math.max(1, parseInt(els.downsample.value||'1', 10));
      const newFs = Math.max(1, Math.floor(state.nativeFs / factor));
      setFs(newFs, true);
    }

    function setFs(val, fromDownsample){
      if(state.nativeFs){ val = Math.min(val, state.nativeFs); }
      val = clamp(parseInt(val||1,10), 1, state.maxFsCap);
      state.currentFs = val;
      if(els.slider){ els.slider.value = String(val); }
      renderFsLabel(val);
      if(!fromDownsample) syncDownsampleFromFs();
      if(cfg.onFsChange) cfg.onFsChange(val);
    }

    function setNativeFs(nativeFs){
      state.nativeFs = nativeFs;
      if(els.slider){ els.slider.max = String(Math.min(state.maxFsCap, nativeFs)); }
      if(!state.currentFs){ setFs(Math.min(state.maxFsCap, nativeFs)); }
      syncDownsampleFromFs();
    }

    const throttledApply = throttle(()=>{ if(state.backend) applyBackendSampling(); }, 300);

    function attachLocalHandlers(){
      if(els.slider){
        els.slider.addEventListener('input', (e)=>{
          setFs(parseInt(e.target.value,10), false);
          if(state.backend && cfg.autoApplyBackend){ throttledApply(); }
        });
        els.slider.addEventListener('change', (e)=>{
          setFs(parseInt(e.target.value,10), false);
          if(state.backend && cfg.autoApplyBackend){ throttledApply(); }
        });
      }
      if(els.downsample){ els.downsample.addEventListener('input', syncFsFromDownsample); }
      if(els.applyBtn && state.backend){ els.applyBtn.addEventListener('click', applyBackendSampling); }
      if(els.resetBtn && state.backend){ els.resetBtn.addEventListener('click', resetBackendSampling); }
    }

    async function applyBackendSampling(){
      if(!state.backend) return;
      const val = parseFloat(els.slider.value);
      try{
        const res = await fetch(state.backend.applyUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ [state.backend.fieldName||'sampling_freq']: val }) });
        const j = await res.json();
        if(j && j.success){ setFs(j.current_sampling ?? val, false); }
        else { /* optionally log */ console.warn('Failed to apply sampling', j); }
      }catch(e){ alert('Failed to apply sampling: ' + (e && e.message ? e.message : String(e))); }
    }

    async function resetBackendSampling(){
      if(!state.backend) return;
      try{
        const res = await fetch(state.backend.resetUrl, { method:'POST' });
        const j = await res.json();
        if(j && j.success){ setFs(j.current_sampling, false); }
        else { /* optionally log */ console.warn('Failed to reset sampling', j); }
      }catch(e){ alert('Failed to reset sampling: ' + (e && e.message ? e.message : String(e))); }
    }

    attachLocalHandlers();

    return {
      setNativeFs,
      setFs,
      getFs: ()=>state.currentFs,
      apply: applyBackendSampling,
      reset: resetBackendSampling
    };
  }

  function createSpeedWindowController(cfg){
    const base = cfg.baseIntervalMs || 100;
    const speedInput = cfg.speedInput;
    const widthInput = cfg.widthInput;
    let speedVal = parseFloat(speedInput && speedInput.value) || 1.0;
    let widthVal = parseFloat(widthInput && widthInput.value) || 5;

    function getDelay(){ return Math.max(20, Math.floor(base / (speedVal || 1))); }

    function onSpeedChange(){ speedVal = parseFloat(speedInput.value)||1; if(cfg.onIntervalChange) cfg.onIntervalChange(getDelay()); }
    function onWidthChange(){ widthVal = parseFloat(widthInput.value)||5; if(cfg.onWidthChange) cfg.onWidthChange(widthVal); }

    if(speedInput){ speedInput.addEventListener('change', onSpeedChange); speedInput.addEventListener('input', onSpeedChange); }
    if(widthInput){ widthInput.addEventListener('change', onWidthChange); }

    return {
      getDelayMs: getDelay,
      getWidth: ()=>widthVal
    };
  }

  function createDisplayManager(cfg){
    const selectEl = cfg.selectEl;
    const containers = cfg.containers || {};
    const sizes = cfg.sizes || {};

    function apply(val){
      Object.keys(containers).forEach(k=>{
        const node = containers[k];
        if(!node) return;
        const plotId = cfg.plotIds && cfg.plotIds[k];
        if(k===val){
          node.style.display = 'block';
          const h = sizes[k];
          if(plotId && h){ try{ Plotly.relayout(plotId, { height: parseInt(h) }); }catch(e){} }
        } else {
          node.style.display = 'none';
        }
      });
      if(cfg.onModeChange) cfg.onModeChange(val);
    }

    if(selectEl){ selectEl.addEventListener('change', (e)=> apply(e.target.value)); apply(selectEl.value); }

    return { apply };
  }

  function createPlotResizer(cfg){
    const ids = cfg.ids || [];
    const relayoutAll = throttle(()=>{
      ids.forEach(id=>{ try{ Plotly.relayout(id, { autosize: true }); }catch(e){} });
    }, 200);
    return { relayoutAll };
  }

  // --- Shared Plotly helpers ---
  function plotNew(plotId, data, layout, config){
    try{ return Plotly.newPlot(plotId, data, layout || {}, config || {}); }catch(e){ /* no-op */ }
  }
  function plotRestyle(plotId, restyleObj, traceIndices){
    try{ return Plotly.restyle(plotId, restyleObj, traceIndices); }catch(e){ /* no-op */ }
  }
  function plotRelayout(plotId, relayoutObj){
    try{ return Plotly.relayout(plotId, relayoutObj); }catch(e){ /* no-op */ }
  }
  function slidingRange(currentTime, widthSec){
    const end = Math.max(0, currentTime||0);
    const start = Math.max(0, end - (widthSec||5));
    return [start, end];
  }

  // --- Shared time-series helpers (EEG-style) ---
  function resampleWithAliasing(arr, outLen){
    const N = (arr && arr.length) || 0;
    if(N <= 1) return N===1 ? [arr[0]] : new Array(Math.max(1, outLen)).fill(0);
    if(outLen <= 1) return [arr[0]];
    const out = new Array(outLen);
    const step = (N - 1) / (outLen - 1);
    for(let i=0;i<outLen;i++){
      const idx = Math.floor(i * step);
      out[i] = (idx>=0 && idx<N) ? arr[idx] : arr[N-1];
    }
    return out;
  }
  function updateBuffer(bufferObj, newSignals, newTime, maxPoints){
    if(!bufferObj) return;
    bufferObj.data = bufferObj.data || [];
    bufferObj.time = bufferObj.time || [];
    if(newSignals && newSignals.length){ bufferObj.data.push(...newSignals); }
    if(newTime && newTime.length){ bufferObj.time.push(...newTime); }
    if(typeof maxPoints === 'number' && maxPoints > 0){
      const extra = bufferObj.data.length - maxPoints;
      if(extra > 0){
        bufferObj.data.splice(0, extra);
        if(bufferObj.time) bufferObj.time.splice(0, extra);
      }
    }
  }

  // --- Stacked time plots (shared between EEG/ECG) ---
  function initStackedTime(containerId, channels, channelNames, widthSec, opts){
    const container = document.getElementById(containerId);
    if(!container) return {};
    while(container.firstChild) container.removeChild(container.firstChild);
    const palette = (opts && opts.colors) || ["#FF5733","#33FF57","#3357FF","#F3FF33","#FF33EC","#33FFF6","#FF8F33","#8F33FF","#33FF99","#FF3333"];
    const lineWidth = (opts && opts.lineWidth) || 1.8;
    const fontColor = (opts && opts.fontColor) || "#e5e7eb";
    const bg = (opts && opts.bg) || "#000";
    const grid = (opts && opts.grid) || "#374151";
    const idMap = {};
    channels.forEach((ch, idx)=>{
      const plotId = `plot-ch-${ch}`;
      const div = document.createElement('div');
      div.id = plotId;
      div.style.width = '100%';
      div.style.height = '200px';
      container.appendChild(div);
      const trace = [{ x: [], y: [], mode: 'lines', name: '', line: { width: lineWidth, color: palette[idx % palette.length] }, showlegend: false, hoverinfo: 'none' }];
      const layout = {
        title: `${channelNames[String(ch)] || ('Ch '+(ch+1))} - Time Graph - Window: ${widthSec}s`,
        paper_bgcolor: bg, plot_bgcolor: bg,
        font: { color: fontColor },
        xaxis: { title: 'Time (s)', color: fontColor, gridcolor: grid, range: [0, widthSec] },
        yaxis: { title: channelNames[String(ch)] || '', color: fontColor, gridcolor: grid, showticklabels: false, autorange: true },
        margin: { t: 40, r: 10, b: 40, l: 40 }
      };
      plotNew(plotId, trace, layout, { responsive: true, displayModeBar: false });
      idMap[ch] = plotId;
    });
    return idMap;
  }

  function updateStackedTime(plotIdMap, buffers, widthSec, currentTime, channelNames){
    const xmin = currentTime - widthSec;
    const xmax = currentTime;
    Object.keys(plotIdMap || {}).forEach(k=>{
      const ch = parseInt(k);
      const pid = plotIdMap[k];
      const buf = buffers[ch];
      if(!buf || !buf.time || !buf.data) return;
      plotRestyle(pid, { x: [buf.time], y: [buf.data] });
      const title = `${channelNames && channelNames[String(ch)] ? channelNames[String(ch)] : ('Ch '+(ch+1))} - Time Graph - Window: ${widthSec}s (Time: ${currentTime.toFixed(2)}s)`;
      plotRelayout(pid, { 'xaxis.range': [xmin, xmax], 'title': title });
    });
  }

  global.createSamplingController = createSamplingController;
  global.createSpeedWindowController = createSpeedWindowController;
  global.createDisplayManager = createDisplayManager;
  global.createPlotResizer = createPlotResizer;
  global.plotNew = plotNew;
  global.plotRestyle = plotRestyle;
  global.plotRelayout = plotRelayout;
  global.slidingRange = slidingRange;
  global.resampleWithAliasing = resampleWithAliasing;
  global.updateBuffer = updateBuffer;
  global.initStackedTime = initStackedTime;
  global.updateStackedTime = updateStackedTime;
  global.initStackedTime = initStackedTime;
  global.updateStackedTime = updateStackedTime;
})(window);
