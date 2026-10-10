/**
 * Voice Agent Client Application
 * Handles WebSocket streaming, Web Audio PCM recording (resampled to 16kHz), and audio playback queue.
 */

// DOM Elements
const connectionBadge = document.getElementById('connectionBadge');
const connectionText = document.getElementById('connectionText');
const chatMessages = document.getElementById('chatMessages');
const clearChatBtn = document.getElementById('clearChatBtn');
const textInput = document.getElementById('textInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const streamingPreview = document.getElementById('streamingPreview');
const previewBadge = document.getElementById('previewBadge');
const previewText = document.getElementById('previewText');
const stateOrb = document.getElementById('stateOrb');
const stateLabel = document.getElementById('stateLabel');
const stateDetail = document.getElementById('stateDetail');
const toggleVoiceBtn = document.getElementById('toggleVoiceBtn');
const voiceBtnText = document.getElementById('voiceBtnText');
const pushTalkBtn = document.getElementById('pushTalkBtn');
const canvas = document.getElementById('audioVisualizer');
const canvasCtx = canvas.getContext('2d');

// State
let ws = null;
let audioContext = null;
let micStream = null;
let micSource = null;
let audioProcessor = null;
let analyser = null;
let isRecording = false;
let isPushToTalk = false;
let audioPlaybackQueue = [];
let isPlayingAudio = false;
let currentAssistantBubble = null;
let currentAudioLevel = 0;

// Initialize Web Audio Visualizer Canvas
function drawVisualizer() {
    requestAnimationFrame(drawVisualizer);
    const width = canvas.width;
    const height = canvas.height;

    canvasCtx.fillStyle = '#0B0F19';
    canvasCtx.fillRect(0, 0, width, height);

    if (!isRecording && !isPlayingAudio) {
        // Draw idle wave line
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#212F4C';
        canvasCtx.beginPath();
        const sliceWidth = width / 64;
        let x = 0;
        for (let i = 0; i < 64; i++) {
            const v = Math.sin(i * 0.2 + Date.now() * 0.003) * 4;
            const y = height / 2 + v;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }
        canvasCtx.stroke();
        return;
    }

    if (analyser) {
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        const barWidth = (width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * height * 0.85;
            const gradient = canvasCtx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#4F46E5');
            gradient.addColorStop(1, '#06B6D4');

            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
    }
}
drawVisualizer();

// WebSocket Setup
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('[WebSocket] Connecting to:', wsUrl);
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('[WebSocket] Connected successfully.');
        connectionBadge.style.color = '#10B981';
        connectionBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        connectionText.textContent = 'Connected';
    };

    ws.onclose = () => {
        console.warn('[WebSocket] Closed. Reconnecting in 2s...');
        connectionBadge.style.color = '#EF4444';
        connectionBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        connectionText.textContent = 'Disconnected (Reconnecting...)';
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error('[WebSocket] Error:', err);
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            try {
                const data = JSON.parse(event.data);
                handleServerEvent(data);
            } catch (e) {
                console.error('Failed to parse JSON:', event.data);
            }
        } else if (event.data instanceof ArrayBuffer) {
            // Received audio chunk (WAV/PCM)
            queueAudioForPlayback(event.data);
        }
    };
}

// Handle Structured Server Events
function handleServerEvent(data) {
    console.log('[Server Event]:', data.type, data);
    switch (data.type) {
        case 'status':
            updateAgentStatus(data.status, data.message);
            break;

        case 'stt_chunk':
            showStreamingPreview('STT', data.transcript);
            break;

        case 'stt_output':
            hideStreamingPreview();
            addUserMessage(data.transcript);
            break;

        case 'agent_chunk':
            showStreamingPreview('Agent', 'Generating response...');
            appendAssistantToken(data.text);
            break;

        case 'agent_complete':
            hideStreamingPreview();
            finalizeAssistantMessage(data.response);
            break;

        case 'error':
            updateAgentStatus('error', data.message || 'Error occurred');
            break;
    }
}

function updateAgentStatus(status, message) {
    stateOrb.className = `state-orb state-${status}`;
    stateLabel.textContent = status.toUpperCase();
    stateDetail.textContent = message || '';
}

function showStreamingPreview(badge, text) {
    streamingPreview.style.display = 'flex';
    previewBadge.textContent = badge;
    previewText.textContent = text;
}

function hideStreamingPreview() {
    streamingPreview.style.display = 'none';
    previewText.textContent = '';
}

// Chat UI Manipulation
function addUserMessage(text) {
    if (!text || !text.trim()) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user';
    msgDiv.innerHTML = `
        <div class="msg-avatar">👤</div>
        <div class="msg-bubble">
            <div class="msg-sender">You</div>
            <div class="msg-text">${escapeHtml(text)}</div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    currentAssistantBubble = null;
}

function appendAssistantToken(token) {
    if (!currentAssistantBubble) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.innerHTML = `
            <div class="msg-avatar">🥪</div>
            <div class="msg-bubble">
                <div class="msg-sender">Voice Assistant</div>
                <div class="msg-text"></div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        currentAssistantBubble = msgDiv.querySelector('.msg-text');
    }
    currentAssistantBubble.textContent += token;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function finalizeAssistantMessage(fullText) {
    if (currentAssistantBubble && fullText) {
        currentAssistantBubble.textContent = fullText;
    }
    currentAssistantBubble = null;
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// Audio Playback Queue
async function queueAudioForPlayback(arrayBuffer) {
    audioPlaybackQueue.push(arrayBuffer);
    if (!isPlayingAudio) {
        playNextAudioChunk();
    }
}

async function playNextAudioChunk() {
    if (audioPlaybackQueue.length === 0) {
        isPlayingAudio = false;
        return;
    }

    isPlayingAudio = true;
    const arrayBuffer = audioPlaybackQueue.shift();

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    try {
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;

        if (!analyser) {
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;
        }
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        source.onended = () => {
            playNextAudioChunk();
        };

        source.start(0);
    } catch (err) {
        console.error('Audio decode/playback error:', err);
        playNextAudioChunk();
    }
}

/**
 * Downsample Float32 audio buffer from inputSampleRate down to targetSampleRate (16000Hz).
 */
function downsampleBuffer(buffer, inputSampleRate, targetSampleRate = 16000) {
    if (inputSampleRate === targetSampleRate) {
        return buffer;
    }
    const sampleRateRatio = inputSampleRate / targetSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0;
        let count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = count > 0 ? accum / count : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

// Web Audio Recording & Streaming (16kHz Mono PCM)
async function startMicrophone() {
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        console.log(`[Audio] AudioContext initialized at sampleRate = ${audioContext.sampleRate}Hz`);

        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });

        micSource = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 64;
        micSource.connect(analyser);

        // Buffer size 4096 gives ~92ms chunks at 44.1/48kHz
        audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        audioProcessor.onaudioprocess = (e) => {
            if (!isRecording && !isPushToTalk) return;

            const inputData = e.inputBuffer.getChannelData(0);
            const inputSampleRate = audioContext.sampleRate;

            // 1. Resample to 16000 Hz PCM
            const resampledData = downsampleBuffer(inputData, inputSampleRate, 16000);

            // 2. Convert Float32 [-1.0, 1.0] to 16-bit PCM (Int16Array)
            const pcm16 = new Int16Array(resampledData.length);
            for (let i = 0; i < resampledData.length; i++) {
                const s = Math.max(-1, Math.min(1, resampledData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            // 3. Send binary PCM16 chunk over WebSocket
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(pcm16.buffer);
            }
        };

        micSource.connect(audioProcessor);
        // Connect to dummy destination so onaudioprocess continues firing
        audioProcessor.connect(audioContext.destination);

        isRecording = true;
        toggleVoiceBtn.classList.add('active');
        voiceBtnText.textContent = 'Stop Microphone';
        updateAgentStatus('listening', 'Microphone active. Say something...');
        console.log('[Audio] Microphone streaming started.');
    } catch (err) {
        console.error('Microphone Access Error:', err);
        alert('Could not access microphone: ' + err.message);
    }
}

function stopMicrophone() {
    isRecording = false;
    toggleVoiceBtn.classList.remove('active');
    voiceBtnText.textContent = 'Start Microphone';

    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }
    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }
    updateAgentStatus('idle', 'Microphone stopped.');
    console.log('[Audio] Microphone streaming stopped.');
}

// Event Listeners
toggleVoiceBtn.addEventListener('click', () => {
    if (isRecording) {
        stopMicrophone();
    } else {
        startMicrophone();
    }
});

// Push-to-Talk Handling
pushTalkBtn.addEventListener('mousedown', async () => {
    isPushToTalk = true;
    pushTalkBtn.classList.add('active');
    if (!isRecording) {
        await startMicrophone();
    }
});

pushTalkBtn.addEventListener('mouseup', () => {
    isPushToTalk = false;
    pushTalkBtn.classList.remove('active');
    stopMicrophone();
});

// Text Input Bar
function sendTextMessage() {
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = '';

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'text_input',
            text: text
        }));
    }
}

sendTextBtn.addEventListener('click', sendTextMessage);
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextMessage();
});

// Quick Prompt Chips
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        if (prompt) {
            textInput.value = prompt;
            sendTextMessage();
        }
    });
});

clearChatBtn.addEventListener('click', () => {
    chatMessages.innerHTML = '';
});

// Initialize WebSocket connection on load
window.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
});
