let audioContext;
let originalBuffer = null;
let resampledBuffer = null;
let originalRate = 44100;
let isPlaying = false;
let currentSourceNode = null;
let uploadedFile = null;

// Recording variables
let mediaRecorder = null;
let audioChunks = [];
let recordingStream = null;
let recordingStartTime = null;
let recordingTimerInterval = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const dragDropArea = document.getElementById('dragDropArea');
const filePrompt = document.getElementById('filePrompt');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const controls = document.getElementById('controls');
const frequencySlider = document.getElementById('samplingFrequency');
const currentFrequencyDisplay = document.getElementById('currentFrequency');
const playOriginalBtn = document.getElementById('playOriginalBtn');
const playSampledBtn = document.getElementById('playSampledBtn');
const playReconstructedBtn = document.getElementById('playReconstructedBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const classifyOriginalBtn = document.getElementById('classifyOriginalBtn');
const classifyResampledBtn = document.getElementById('classifyResampledBtn');
const classifyReconstructedBtn = document.getElementById('classifyReconstructedBtn');
const antiAliasingToggle = document.getElementById('antiAliasingToggle');
const messageBox = document.getElementById('messageBox');

// Results elements
const originalResults = document.getElementById('originalResults');
const originalGender = document.getElementById('originalGender');
const originalConfidence = document.getElementById('originalConfidence');
const originalPitch = document.getElementById('originalPitch');
const originalSampleRate = document.getElementById('originalSampleRate');
const originalAccentBar = document.getElementById('originalAccentBar');

const resampledResults = document.getElementById('resampledResults');
const resampledGender = document.getElementById('resampledGender');
const resampledConfidence = document.getElementById('resampledConfidence');
const resampledPitch = document.getElementById('resampledPitch');
const resampledSampleRate = document.getElementById('resampledSampleRate');
const resampledAccentBar = document.getElementById('resampledAccentBar');

const reconstructedResults = document.getElementById('reconstructedResults');
const reconstructedGender = document.getElementById('reconstructedGender');
const reconstructedConfidence = document.getElementById('reconstructedConfidence');
const reconstructedPitch = document.getElementById('reconstructedPitch');
const reconstructedSampleRate = document.getElementById('reconstructedSampleRate');
const reconstructedAccentBar = document.getElementById('reconstructedAccentBar');

const aliasingAnalysis = document.getElementById('aliasingAnalysis');
const classificationMatch = document.getElementById('classificationMatch');
const confidenceChange = document.getElementById('confidenceChange');
const pitchChange = document.getElementById('pitchChange');
const aliasingEffect = document.getElementById('aliasingEffect');

const recordBtn = document.getElementById('recordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const recordingTimer = document.getElementById('recordingTimer');

// Store classification results
let originalClassification = null;
let resampledClassification = null;
let reconstructedClassification = null;

// Initialize AudioContext
function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        frequencySlider.max = audioContext.sampleRate;
        frequencySlider.value = audioContext.sampleRate;
        currentFrequencyDisplay.textContent = audioContext.sampleRate;
        messageBox.textContent = `Native Sample Rate: ${audioContext.sampleRate} Hz`;
        originalRate = audioContext.sampleRate;
    }
}

// Audio Loading
function decodeAudio(file) {
    uploadedFile = file;
    initAudioContext();
    const reader = new FileReader();

    reader.onload = function(event) {
        const arrayBuffer = event.target.result;
        audioContext.decodeAudioData(arrayBuffer, 
            function(buffer) {
                originalBuffer = buffer;
                originalRate = buffer.sampleRate;
                frequencySlider.max = originalRate;
                frequencySlider.value = originalRate;
                currentFrequencyDisplay.textContent = originalRate;
                
                resampleAudio(originalRate);
                updateUI(file.name);
            },
            function(e) {
                messageBox.textContent = "Error decoding audio data.";
                console.error("Audio decoding error:", e);
            }
        );
    };

    reader.onerror = function() {
        messageBox.textContent = "Error reading file.";
    };

    reader.readAsArrayBuffer(file);
}

function updateUI(fileName) {
    controls.classList.remove('opacity-50', 'pointer-events-none');
    playOriginalBtn.disabled = false;
    playSampledBtn.disabled = false;
    playReconstructedBtn.disabled = false;
    resetBtn.disabled = false;
    classifyOriginalBtn.disabled = false;
    classifyResampledBtn.disabled = false;
    classifyReconstructedBtn.disabled = false;
    fileNameDisplay.textContent = `Loaded: ${fileName}`;
    fileNameDisplay.classList.remove('hidden');
    filePrompt.classList.add('hidden');
    messageBox.textContent = `File ready. Original rate: ${originalRate} Hz`;
}

// Cubic interpolation helper function (Catmull-Rom Spline)
function cubicInterpolate(data, index) {
    const i = Math.floor(index);
    const fraction = index - i;
    
    // Get 4 points for cubic interpolation
    const p0 = data[Math.max(0, i - 1)] || 0;
    const p1 = data[i] || 0;
    const p2 = data[Math.min(i + 1, data.length - 1)] || 0;
    const p3 = data[Math.min(i + 2, data.length - 1)] || 0;
    
    // Catmull-Rom spline interpolation
    const a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const a2 = -0.5 * p0 + 0.5 * p2;
    const a3 = p1;
    
    return a0 * fraction * fraction * fraction + 
           a1 * fraction * fraction + 
           a2 * fraction + 
           a3;
}

// Simple resampling without anti-aliasing filter
function resampleAudio(targetRate) {
    if (!originalBuffer) return;

    const ratio = originalRate / targetRate;
    const duration = originalBuffer.length / originalRate;
    const downsampledLength = Math.floor(duration * targetRate);

    const tempDownsampled = [];
    
    for (let channel = 0; channel < originalBuffer.numberOfChannels; channel++) {
        const originalData = originalBuffer.getChannelData(channel);
        const channelDownsampled = new Float32Array(downsampledLength);
        
        // Use linear interpolation for downsampling
        for (let i = 0; i < downsampledLength; i++) {
            const exactIndex = i * ratio;
            const lowerIndex = Math.floor(exactIndex);
            const upperIndex = Math.min(lowerIndex + 1, originalData.length - 1);
            const fraction = exactIndex - lowerIndex;
            
            channelDownsampled[i] = originalData[lowerIndex] * (1 - fraction) + 
                                   originalData[upperIndex] * fraction;
        }
        tempDownsampled.push(channelDownsampled);
    }

    // Create resampled buffer at original sample rate for playback
    resampledBuffer = audioContext.createBuffer(
        originalBuffer.numberOfChannels,
        originalBuffer.length,
        originalBuffer.sampleRate
    );

    for (let channel = 0; channel < originalBuffer.numberOfChannels; channel++) {
        const downsampledData = tempDownsampled[channel];
        const resampledData = resampledBuffer.getChannelData(channel);
        
        // Use cubic interpolation for upsampling back to original rate
        for (let i = 0; i < resampledData.length; i++) {
            const downsampledIndex = i / ratio;
            resampledData[i] = cubicInterpolate(downsampledData, downsampledIndex);
        }
    }
}

// Classification Functions
async function classifyOriginal() {
    if (!uploadedFile) {
        messageBox.textContent = "No file uploaded!";
        return;
    }

    classifyOriginalBtn.disabled = true;
    messageBox.textContent = "Classifying original audio...";

    const formData = new FormData();
    formData.append('audio', uploadedFile);
    formData.append('use_antialiasing', 'false');

    try {
        const response = await fetch('/voice/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Classification failed');
        }
        
        originalClassification = {
            gender: data.gender,
            confidence: data.confidence,
            pitch: data.pitch,
            sampleRate: originalRate
        };
        
        displayOriginalResults(originalClassification);
        messageBox.textContent = "Original audio classified successfully!";
        
        if (reconstructedClassification) {
            displayMLAnalysis();
        }
    } catch (error) {
        messageBox.textContent = "Error: " + error.message;
        console.error('Classification error:', error);
    } finally {
        classifyOriginalBtn.disabled = false;
    }
}

async function classifyResampled() {
    if (!resampledBuffer) {
        messageBox.textContent = "No resampled audio available!";
        return;
    }

    classifyResampledBtn.disabled = true;
    messageBox.textContent = "Classifying resampled audio (raw undersampled)...";

    try {
        const wavBlob = audioBufferToWav(resampledBuffer);
        const wavFile = new File([wavBlob], 'resampled_audio.wav', { type: 'audio/wav' });
        
        const formData = new FormData();
        formData.append('audio', wavFile);
        formData.append('use_antialiasing', 'false');
        const targetRate = parseInt(frequencySlider.value);
        formData.append('effective_sr', targetRate.toString());

        const response = await fetch('/voice/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Classification failed');
        }
        
        resampledClassification = {
            gender: data.gender,
            confidence: data.confidence,
            pitch: data.pitch,
            sampleRate: targetRate
        };
        
        displayResampledResults(resampledClassification);
        messageBox.textContent = "Raw undersampled audio classified successfully!";
    } catch (error) {
        messageBox.textContent = "Error: " + error.message;
        console.error('Classification error:', error);
    } finally {
        classifyResampledBtn.disabled = false;
    }
}

async function classifyReconstructed() {
    if (!resampledBuffer) {
        messageBox.textContent = "Please resample audio first!";
        return;
    }

    const useMLModel = antiAliasingToggle.checked;

    classifyReconstructedBtn.disabled = true;
    
    if (useMLModel) {
        messageBox.textContent = "Applying ML-based anti-aliasing reconstruction...";
    } else {
        messageBox.textContent = "Classifying raw undersampled audio...";
    }

    try {
        const wavBlob = audioBufferToWav(resampledBuffer);
        const wavFile = new File([wavBlob], 'resampled_audio.wav', { type: 'audio/wav' });
        
        const formData = new FormData();
        formData.append('audio', wavFile);
        formData.append('use_antialiasing', useMLModel ? 'true' : 'false');
        
        const targetRate = parseInt(frequencySlider.value);
        if (!useMLModel) {
            formData.append('effective_sr', targetRate.toString());
        }

        const response = await fetch('/voice/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Classification failed');
        }
        
        reconstructedClassification = {
            gender: data.gender,
            confidence: data.confidence,
            pitch: data.pitch,
            sampleRate: useMLModel ? 16000 : targetRate,
            downsampledFrom: targetRate,
            mlEnhanced: data.antialiasing_applied || false
        };
        
        displayReconstructedResults(reconstructedClassification);
        
        if (useMLModel) {
            messageBox.textContent = "ML-enhanced classification complete!";
        } else {
            messageBox.textContent = "Raw undersampled classification complete!";
        }
        
        if (originalClassification) {
            displayMLAnalysis();
        }
    } catch (error) {
        messageBox.textContent = "Error: " + error.message;
        console.error('Classification error:', error);
    } finally {
        classifyReconstructedBtn.disabled = false;
    }
}

// Display Functions
function displayOriginalResults(data) {
    originalGender.textContent = data.gender.toUpperCase();
    originalConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
    originalPitch.textContent = data.pitch.toFixed(1) + ' Hz';
    originalSampleRate.textContent = data.sampleRate;

    originalResults.classList.remove('results-male', 'results-female');
    originalAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');

    if (data.gender === 'male') {
        originalResults.classList.add('results-male');
        originalAccentBar.classList.add('gender-accent-bar-male');
        originalGender.className = 'text-2xl font-bold text-blue-400';
    } else {
        originalResults.classList.add('results-female');
        originalAccentBar.classList.add('gender-accent-bar-female');
        originalGender.className = 'text-2xl font-bold text-pink-400';
    }

    originalResults.classList.remove('hidden');
}

function displayResampledResults(data) {
    resampledGender.textContent = data.gender.toUpperCase();
    resampledConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
    resampledPitch.textContent = data.pitch.toFixed(1) + ' Hz';
    resampledSampleRate.textContent = data.sampleRate;

    resampledResults.classList.remove('results-male', 'results-female');
    resampledAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');

    if (data.gender === 'male') {
        resampledResults.classList.add('results-male');
        resampledAccentBar.classList.add('gender-accent-bar-male');
        resampledGender.className = 'text-2xl font-bold text-blue-400';
    } else {
        resampledResults.classList.add('results-female');
        resampledAccentBar.classList.add('gender-accent-bar-female');
        resampledGender.className = 'text-2xl font-bold text-pink-400';
    }

    resampledResults.classList.remove('hidden');
}

function displayReconstructedResults(data) {
    reconstructedGender.textContent = data.gender.toUpperCase();
    reconstructedConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
    reconstructedPitch.textContent = data.pitch.toFixed(1) + ' Hz';
    
    const sampleRateText = data.mlEnhanced ? '16000 (ML Enhanced)' : data.sampleRate;
    reconstructedSampleRate.textContent = sampleRateText;

    reconstructedResults.classList.remove('results-male', 'results-female');
    reconstructedAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');

    if (data.gender === 'male') {
        reconstructedResults.classList.add('results-male');
        reconstructedAccentBar.classList.add('gender-accent-bar-male');
        reconstructedGender.className = 'text-2xl font-bold text-blue-400';
    } else {
        reconstructedResults.classList.add('results-female');
        reconstructedAccentBar.classList.add('gender-accent-bar-female');
        reconstructedGender.className = 'text-2xl font-bold text-pink-400';
    }

    reconstructedResults.classList.remove('hidden');
}

function displayMLAnalysis() {
    if (!originalClassification || !reconstructedClassification) return;

    const match = originalClassification.gender === reconstructedClassification.gender;
    const confChange = ((reconstructedClassification.confidence - originalClassification.confidence) * 100).toFixed(1);
    const pitchDiff = (reconstructedClassification.pitch - originalClassification.pitch).toFixed(1);

    classificationMatch.textContent = match ? '✓ MATCH' : '✗ DIFFERENT';
    classificationMatch.className = match ? 'font-bold text-lg text-green-400' : 'font-bold text-lg text-yellow-400';

    confidenceChange.textContent = (confChange >= 0 ? '+' : '') + confChange + '%';
    confidenceChange.className = confChange >= 0 ? 'font-bold text-lg text-green-400' : 'font-bold text-lg text-red-400';

    pitchChange.textContent = (pitchDiff >= 0 ? '+' : '') + pitchDiff + ' Hz';
    pitchChange.className = 'font-bold text-lg text-cyan-400';

    let effect = '';
    if (reconstructedClassification.mlEnhanced) {
        if (match && confChange >= 0) {
            effect = '✓ ML ENHANCEMENT SUCCESSFUL: Neural network reconstructed audio quality. Classification improved.';
        } else if (match) {
            effect = '✓ ML ENHANCEMENT EFFECTIVE: Neural network maintained classification accuracy.';
        } else {
            effect = '⚠️ ML RECONSTRUCTION CHANGED RESULT: Neural network altered audio characteristics.';
        }
    } else {
        effect = '⚠️ RAW UNDERSAMPLED SIGNAL: No ML enhancement applied. This shows the effects of aliasing.';
    }

    aliasingEffect.textContent = effect;
    aliasingAnalysis.classList.remove('hidden');
}

// Playback and utility functions
function stopAudio() {
    if (currentSourceNode) {
        currentSourceNode.stop();
        currentSourceNode = null;
    }
    isPlaying = false;
    if (originalBuffer) {
        playOriginalBtn.disabled = false;
        playSampledBtn.disabled = false;
         playReconstructedBtn.disabled = false;
        classifyOriginalBtn.disabled = false;
        classifyResampledBtn.disabled = false;
        classifyReconstructedBtn.disabled = false;
    }
    pauseBtn.disabled = true;
    messageBox.textContent = "Playback stopped.";
}

function playAudio(buffer) {
    if (isPlaying) stopAudio();
    initAudioContext();
    
    currentSourceNode = audioContext.createBufferSource();
    currentSourceNode.buffer = buffer;
    currentSourceNode.connect(audioContext.destination);
    
    currentSourceNode.onended = () => {
        if (isPlaying) {
            isPlaying = false;
            currentSourceNode = null;
            playOriginalBtn.disabled = false;
            playSampledBtn.disabled = false;
             playReconstructedBtn.disabled = false;
            pauseBtn.disabled = true;
            classifyOriginalBtn.disabled = false;
            classifyResampledBtn.disabled = false;
            classifyReconstructedBtn.disabled = false;
            messageBox.textContent = "Playback finished.";
        }
    };

    currentSourceNode.start(0);
    isPlaying = true;
    playOriginalBtn.disabled = true;
    playSampledBtn.disabled = true;
      playReconstructedBtn.disabled = true;
    classifyOriginalBtn.disabled = true;
    classifyResampledBtn.disabled = true;
    classifyReconstructedBtn.disabled = true;
    pauseBtn.disabled = false;
    messageBox.textContent = "Playing audio...";
}

function resetApp() {
    stopAudio();
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (recordingStream) {
        recordingStream.getTracks().forEach(track => track.stop());
        recordingStream = null;
    }
    if (recordingTimerInterval) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
    }
    
    originalBuffer = null;
    resampledBuffer = null;
    uploadedFile = null;
    isPlaying = false;
    currentSourceNode = null;
    audioChunks = [];
    recordingStartTime = null;
    originalClassification = null;
    resampledClassification = null;
    reconstructedClassification = null;
    
    recordBtn.disabled = false;
    recordBtn.classList.remove('record-btn-active');
    stopRecordBtn.disabled = true;
    recordingTimer.classList.add('hidden');
    recordingStatus.textContent = '';
    recordingStatus.classList.remove('recording-indicator');
    
    controls.classList.add('opacity-50', 'pointer-events-none');
    playOriginalBtn.disabled = true;
    playSampledBtn.disabled = true;
     playReconstructedBtn.disabled = true;
    pauseBtn.disabled = true;
    resetBtn.disabled = true;
    classifyOriginalBtn.disabled = true;
    classifyResampledBtn.disabled = true;
    classifyReconstructedBtn.disabled = true;
    
    fileNameDisplay.classList.add('hidden');
    filePrompt.classList.remove('hidden');
    
    const defaultRate = audioContext ? audioContext.sampleRate : 44100;
    frequencySlider.min = 4000;
    frequencySlider.max = defaultRate;
    frequencySlider.value = defaultRate;
    currentFrequencyDisplay.textContent = `${defaultRate} (Original Rate)`;
    
    originalResults.classList.add('hidden');
    resampledResults.classList.add('hidden');
    reconstructedResults.classList.add('hidden');
    aliasingAnalysis.classList.add('hidden');
    
    originalResults.classList.remove('results-male', 'results-female');
    resampledResults.classList.remove('results-male', 'results-female');
    reconstructedResults.classList.remove('results-male', 'results-female');
    originalAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');
    resampledAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');
    reconstructedAccentBar.classList.remove('gender-accent-bar-male', 'gender-accent-bar-female');

    messageBox.textContent = "Application reset.";
}

// Recording Functions
async function startRecording() {
    try {
        recordingStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });

        audioChunks = [];
        const options = { mimeType: 'audio/webm' };
        mediaRecorder = new MediaRecorder(recordingStream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await convertToWavAndLoad(audioBlob);
            
            if (recordingStream) {
                recordingStream.getTracks().forEach(track => track.stop());
                recordingStream = null;
            }
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();
        
        recordBtn.disabled = true;
        recordBtn.classList.add('record-btn-active');
        stopRecordBtn.disabled = false;
        recordingTimer.classList.remove('hidden');
        recordingStatus.textContent = '🔴 Recording...';
        recordingStatus.classList.add('recording-indicator');
        
        recordingTimerInterval = setInterval(updateRecordingTimer, 100);
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        recordingStatus.textContent = 'Error: Microphone access denied.';
        recordingStatus.classList.remove('recording-indicator');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        recordBtn.disabled = false;
        recordBtn.classList.remove('record-btn-active');
        stopRecordBtn.disabled = true;
        recordingTimer.classList.add('hidden');
        recordingStatus.textContent = 'Processing recording...';
        recordingStatus.classList.remove('recording-indicator');
        
        if (recordingTimerInterval) {
            clearInterval(recordingTimerInterval);
            recordingTimerInterval = null;
        }
    }
}

function updateRecordingTimer() {
    if (recordingStartTime) {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
}

async function convertToWavAndLoad(webmBlob) {
    try {
        initAudioContext();
        const arrayBuffer = await webmBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        const wavFile = new File([wavBlob], 'recorded_voice.wav', { type: 'audio/wav' });
        
        uploadedFile = wavFile;
        decodeAudio(wavFile);
        
        recordingStatus.textContent = '✓ Recording saved!';
        setTimeout(() => {
            recordingStatus.textContent = '';
        }, 3000);
        
    } catch (error) {
        console.error('Error converting recording:', error);
        recordingStatus.textContent = 'Error processing recording.';
    }
}

function audioBufferToWav(buffer) {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let pos = 0;

    const setUint16 = (data) => {
        view.setUint16(pos, data, true);
        pos += 2;
    };
    const setUint32 = (data) => {
        view.setUint32(pos, data, true);
        pos += 4;
    };

    setUint32(0x46464952);
    setUint32(length - 8);
    setUint32(0x45564157);
    setUint32(0x20746d66);
    setUint32(16);
    setUint16(1);
    setUint16(buffer.numberOfChannels);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * buffer.numberOfChannels * 2);
    setUint16(buffer.numberOfChannels * 2);
    setUint16(16);
    setUint32(0x61746164);
    setUint32(length - pos - 4);

    for (let i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    let offset = 0;
    while (pos < length) {
        for (let i = 0; i < buffer.numberOfChannels; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

// Event Listeners
recordBtn.addEventListener('click', startRecording);
stopRecordBtn.addEventListener('click', stopRecording);

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        decodeAudio(e.target.files[0]);
    }
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, () => {
        dragDropArea.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dragDropArea.addEventListener(eventName, () => {
        dragDropArea.classList.remove('drag-over');
    }, false);
});

dragDropArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('audio/')) {
        decodeAudio(files[0]);
    } else {
        messageBox.textContent = "Please drop a valid audio file.";
    }
}, false);

frequencySlider.addEventListener('input', (e) => {
    const targetRate = parseInt(e.target.value, 10);
    currentFrequencyDisplay.textContent = targetRate;
    
    if (targetRate === originalRate) {
        currentFrequencyDisplay.textContent += ' (Original Rate)';
    } else if (targetRate < 8000) {
        currentFrequencyDisplay.textContent += ' (Severe Degradation)';
    }
    
    if (originalBuffer) {
        resampleAudio(targetRate);
    }
});

playOriginalBtn.addEventListener('click', () => {
    if (originalBuffer) playAudio(originalBuffer);
});

playSampledBtn.addEventListener('click', () => {
    if (resampledBuffer) playAudio(resampledBuffer);
});


playReconstructedBtn.addEventListener('click', () => {
    if (resampledBuffer) playAudio(resampledBuffer);
});

pauseBtn.addEventListener('click', stopAudio);
resetBtn.addEventListener('click', resetApp);
classifyOriginalBtn.addEventListener('click', classifyOriginal);
classifyResampledBtn.addEventListener('click', classifyResampled);
classifyReconstructedBtn.addEventListener('click', classifyReconstructed);

antiAliasingToggle.addEventListener('change', () => {
    const isEnabled = antiAliasingToggle.checked;
    messageBox.textContent = isEnabled ? 
        "ML Anti-Aliasing enabled: Will use neural network for reconstruction." : 
        "ML Anti-Aliasing disabled: Will use raw undersampled signal.";
});

// Initialize on load
document.addEventListener('DOMContentLoaded', initAudioContext);