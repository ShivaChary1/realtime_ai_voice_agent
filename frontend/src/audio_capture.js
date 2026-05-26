/**
 * frontend/src/audio_capture.js
 *
 * Handles:
 *  - Microphone access with echo cancellation
 *  - AudioWorklet-based PCM extraction (Float32 → Int16 @ 16kHz)
 *  - Client-side VAD via @ricky0123/vad-web
 *  - WebSocket streaming to backend
 *  - Barge-in detection
 *  - Auto-reconnect with exponential backoff
 */

import { MicVAD } from "@ricky0123/vad-web";

// ── Config ────────────────────────────────────────────────────────────────────
const CONFIG = {
  SAMPLE_RATE:          16000,
  CHANNELS:             1,
  CHUNK_INTERVAL_MS:    250,            // send chunks every 250ms
  SILENCE_PADDING_MS:   300,            // added server-side, not here
  WS_URL:               `ws://${window.location.hostname}:8000/ws`,
  MAX_RECONNECT_ATTEMPTS: 5,
  RECONNECT_BASE_MS:    500,
};

// ── State ─────────────────────────────────────────────────────────────────────
let socket         = null;
let sessionId      = null;
let currentTurnId  = null;
let chunkIndex     = 0;
let reconnectCount = 0;
let isAgentSpeaking = false;
let audioContext   = null;
let vadInstance    = null;
let micStream      = null;
let speechBuffer   = [];   // accumulate Float32 frames during speech
let isRecording    = false;

// ── Session ID ────────────────────────────────────────────────────────────────

function getOrCreateSessionId() {
  let id = sessionStorage.getItem("voice_session_id");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("voice_session_id", id);
  }
  return id;
}

// ── PCM Conversion ────────────────────────────────────────────────────────────

/**
 * Convert Float32Array (Web Audio API output) to Int16 PCM bytes.
 * This is the format Whisper and most STT APIs expect.
 */
function float32ToInt16(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    // Clamp to [-1, 1] then scale to Int16 range [-32768, 32767]
    const clamped = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = clamped < 0
      ? clamped * 32768
      : clamped * 32767;
  }
  return int16Array.buffer;
}

/**
 * Resample audio from source sample rate to 16kHz.
 * Required if the browser AudioContext runs at 48kHz (common default).
 */
async function resampleTo16kHz(audioData, sourceSampleRate) {
  if (sourceSampleRate === CONFIG.SAMPLE_RATE) return audioData;

  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(audioData.length * CONFIG.SAMPLE_RATE / sourceSampleRate),
    CONFIG.SAMPLE_RATE,
  );
  const buffer = offlineCtx.createBuffer(1, audioData.length, sourceSampleRate);
  buffer.copyToChannel(audioData, 0);

  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineCtx.destination);
  source.start();

  const rendered = await offlineCtx.startRendering();
  return rendered.getChannelData(0);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWebSocket() {
  sessionId = getOrCreateSessionId();
  const url = `${CONFIG.WS_URL}/${sessionId}`;

  console.log(`[WS] Connecting to ${url}`);
  updateStatus("connecting");

  socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    console.log("[WS] Connected");
    reconnectCount = 0;
    updateStatus("connected");

    // Send session_start
    sendJSON({
      type: "session_start",
      session_id: sessionId,
      timestamp: Date.now(),
    });
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleServerMessage(JSON.parse(event.data));
    }
  };

  socket.onclose = (event) => {
    console.warn(`[WS] Closed: code=${event.code} reason=${event.reason}`);
    updateStatus("disconnected");
    scheduleReconnect();
  };

  socket.onerror = (error) => {
    console.error("[WS] Error:", error);
  };
}

function scheduleReconnect() {
  if (reconnectCount >= CONFIG.MAX_RECONNECT_ATTEMPTS) {
    console.error("[WS] Max reconnect attempts reached.");
    updateStatus("failed");
    return;
  }

  const delay = CONFIG.RECONNECT_BASE_MS * Math.pow(2, reconnectCount);
  reconnectCount++;
  console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectCount})`);
  updateStatus("reconnecting");
  setTimeout(connectWebSocket, delay);
}

function sendJSON(obj) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(obj));
  }
}

function sendBinary(arrayBuffer) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(arrayBuffer);
  }
}

// ── Server message handler ────────────────────────────────────────────────────

function handleServerMessage(msg) {
  console.debug("[SERVER]", msg);

  switch (msg.type) {
    case "session_ack":
      console.log(`[SESSION] ${msg.status}: ${msg.session_id}`);
      break;

    case "transcript":
      onTranscript?.(msg.text, msg.language);
      break;

    case "agent_text":
      onAgentText?.(msg.text, msg.is_partial);
      break;

    case "audio_response":
      // Stage 6: play TTS audio
      isAgentSpeaking = true;
      onAudioResponse?.(msg);
      break;

    case "stop_playback":
      // Barge-in: stop whatever audio is playing
      isAgentSpeaking = false;
      onStopPlayback?.();
      break;

    case "latency_report":
      console.info("[LATENCY]", msg.durations_ms, `total=${msg.total_ms}ms`);
      onLatencyReport?.(msg);
      break;

    case "error":
      console.error(`[ERROR] ${msg.code}: ${msg.message}`);
      onError?.(msg);
      break;

    case "pong":
      // Could calculate RTT here
      break;
  }
}

// ── VAD + Audio Capture ───────────────────────────────────────────────────────

async function startCapture() {
  try {
    // Request microphone with quality constraints
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation:   true,
        noiseSuppression:   true,
        autoGainControl:    true,
        channelCount:       CONFIG.CHANNELS,
        sampleRate:         CONFIG.SAMPLE_RATE,
      },
    });

    console.log("[MIC] Microphone access granted");

    // Initialize VAD
    vadInstance = await MicVAD.new({
      stream: micStream,
      positiveSpeechThreshold: 0.8,     // higher = less sensitive
      negativeSpeechThreshold: 0.5,
      minSpeechFrames: 5,
      preSpeechPadFrames: 10,

      onSpeechStart: () => {
        console.log("[VAD] Speech started");
        currentTurnId = crypto.randomUUID();
        chunkIndex    = 0;
        speechBuffer  = [];
        isRecording   = true;
        updateStatus("recording");

        // Barge-in: if agent is speaking, interrupt it
        if (isAgentSpeaking) {
          console.log("[BARGE-IN] Detected — sending interrupt");
          sendJSON({
            type:       "barge_in",
            session_id: sessionId,
            turn_id:    currentTurnId,
            timestamp:  Date.now(),
          });
          isAgentSpeaking = false;
          onStopPlayback?.();
        }
      },

      onFrameProcessed: (probabilities) => {
        // Real-time frame - not used for audio, just monitoring
      },

      onSpeechEnd: async (rawAudio) => {
        // rawAudio is Float32Array at 16kHz from VAD model
        console.log("[VAD] Speech ended — processing audio");
        isRecording = false;
        updateStatus("processing");

        const speechEndTs = Date.now();

        // Resample if needed (vad-web typically outputs 16kHz already)
        const resampled = await resampleTo16kHz(rawAudio, 16000);

        // Notify backend of speech end (with client-side timestamp for latency)
        sendJSON({
          type:       "speech_end",
          session_id: sessionId,
          turn_id:    currentTurnId,
          timestamp:  speechEndTs,
        });

        // Send PCM audio as binary
        const pcmBuffer = float32ToInt16(resampled);
        sendBinary(pcmBuffer);

        console.log(
          `[AUDIO] Sent: turn=${currentTurnId} ` +
          `samples=${resampled.length} bytes=${pcmBuffer.byteLength}`
        );

        updateStatus("connected");
      },

      onVADMisfire: () => {
        console.debug("[VAD] Misfire — discarding");
        speechBuffer  = [];
        isRecording   = false;
        updateStatus("connected");
      },
    });

    vadInstance.start();
    console.log("[VAD] Started");

  } catch (err) {
    if (err.name === "NotAllowedError") {
      console.error("[MIC] Permission denied");
      updateStatus("mic_denied");
      onError?.({ code: "mic_denied", message: "Microphone permission denied." });
    } else if (err.name === "NotFoundError") {
      console.error("[MIC] No microphone found");
      updateStatus("mic_not_found");
      onError?.({ code: "mic_not_found", message: "No microphone found." });
    } else {
      console.error("[MIC] Error:", err);
    }
  }
}

function stopCapture() {
  vadInstance?.destroy();
  micStream?.getTracks().forEach((t) => t.stop());
  sendJSON({
    type:       "session_end",
    session_id: sessionId,
    timestamp:  Date.now(),
  });
  socket?.close(1000, "User stopped");
  updateStatus("idle");
  console.log("[CAPTURE] Stopped");
}

// ── Status UI helper ──────────────────────────────────────────────────────────

function updateStatus(state) {
  const el = document.getElementById("connection-status");
  if (!el) return;

  const labels = {
    idle:         "⚪ Idle",
    connecting:   "🟡 Connecting...",
    connected:    "🟢 Connected",
    recording:    "🔴 Listening...",
    processing:   "🔵 Processing...",
    reconnecting: "🟠 Reconnecting...",
    disconnected: "⚪ Disconnected",
    failed:       "🔴 Connection Failed",
    mic_denied:   "🔴 Mic Permission Denied",
    mic_not_found:"🔴 No Microphone Found",
  };
  el.textContent = labels[state] || state;
}

// ── Callback hooks (set by UI layer) ─────────────────────────────────────────
// Assign these from your UI code:
//   import * as capture from './audio_capture.js'
//   capture.onTranscript = (text, lang) => { ... }

export let onTranscript    = null;
export let onAgentText     = null;
export let onAudioResponse = null;
export let onStopPlayback  = null;
export let onLatencyReport = null;
export let onError         = null;

// ── Public API ────────────────────────────────────────────────────────────────

export { connectWebSocket, startCapture, stopCapture };