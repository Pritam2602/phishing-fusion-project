// src/App.jsx
import React, { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [useRecording, setUseRecording] = useState(true);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);

  // CONTEXT STATE
  const [senderId, setSenderId] = useState("");
  const [isShortCode, setIsShortCode] = useState(false);
  const [hasUrl, setHasUrl] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  // START RECORDING
  const startRecording = async () => {
    try {
      setError("");
      setResult(null);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream); // don't force mimeType

      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setRecordedBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setRecording(true);
      setRecordedBlob(null);
    } catch (err) {
      setError("Microphone permission denied or recording not supported.");
    }
  };

  // STOP RECORDING
  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecording(false);
    }
  };

  // SUBMIT (Prediction)
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    const hasText = text.trim().length > 0;
    const hasAudio = useRecording ? !!recordedBlob : !!file;

    if (!hasText && !hasAudio) {
      setError("Please provide text and/or audio.");
      return;
    }

    const formData = new FormData();
    formData.append("text", hasText ? text : "");

    let url = "";

    // Recorded audio -> predict_recorded (we send recorded blob as audio_wav)
    // Add Context params
    formData.append("sender_id", senderId);
    formData.append("is_short_code", isShortCode);
    formData.append("has_url", hasUrl);

    if (useRecording && recordedBlob) {
      // name extension based on mime type: audio/webm etc
      const ext = (recordedBlob.type && recordedBlob.type.split("/")[1]) || "webm";
      formData.append("audio_wav", recordedBlob, `recording.${ext}`);
      url = `${API_BASE}/predict_recorded`;
    } else if (!useRecording && file) {
      // user uploaded .npy mel file
      formData.append("audio_file", file);
      url = `${API_BASE}/predict`;
    } else {
      url = `${API_BASE}/predict`;
    }

    try {
      setLoading(true);

      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Request failed");
      }

      const data = await res.json();

      // NEW: if backend returned a transcription, auto-fill the text field
      if (data.transcription) {
        setText(data.transcription);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <div className="card">
        <h1>Phishing Fusion Demo</h1>
        <p className="subtitle">
          Enter text and optionally add audio (record or .npy). Works with text-only, audio-only, or both (fusion).
        </p>

        <form className="form" onSubmit={handleSubmit}>
          {/* TEXT INPUT */}
          <label className="field">
            <span>Text Message</span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Your account has been compromised..."
              rows={4}
            />
          </label>

          {/* CONTEXT INPUTS */}
          <div className="field-group">
            <label className="field">
              <span>Sender ID (Optional)</span>
              <input
                type="text"
                value={senderId}
                onChange={(e) => setSenderId(e.target.value)}
                placeholder="e.g. VM-HDFCBK or +9198..."
                style={{ padding: "0.6rem", borderRadius: 6, border: "1px solid #444", background: "rgba(255,255,255,0.05)", color: "#fff" }}
              />
            </label>

            <div className="checkbox-row" style={{ display: "flex", gap: "1.5rem", marginTop: "0.5rem" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={isShortCode}
                  onChange={(e) => setIsShortCode(e.target.checked)}
                />
                Is Short Code?
              </label>

              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={hasUrl}
                  onChange={(e) => setHasUrl(e.target.checked)}
                />
                Contains URL?
              </label>
            </div>
          </div>

          {/* TRANSCRIPTION DISPLAY */}
          {result && result.transcription && (
            <div className="field">
              <span>Transcribed Audio</span>
              <div style={{ background: "rgba(255,255,255,0.03)", padding: "0.6rem", borderRadius: 8 }}>
                {result.transcription}
              </div>
            </div>
          )}

          {/* AUDIO MODE */}
          <div className="field">
            <span>Audio Source</span>
            <div className="toggle-row">
              <button type="button" className={useRecording ? "toggle active" : "toggle"} onClick={() => setUseRecording(true)}>
                Record with microphone
              </button>
              <button type="button" className={!useRecording ? "toggle active" : "toggle"} onClick={() => setUseRecording(false)}>
                Upload .npy file
              </button>
            </div>
          </div>

          {/* RECORD UI */}
          {useRecording ? (
            <div className="field">
              <span>Recorded Audio</span>
              <div className="record-row">
                {!recording ? (
                  <button type="button" className="record-btn" onClick={startRecording} disabled={loading}>
                    {recordedBlob ? "Re-record" : "Start Recording"}
                  </button>
                ) : (
                  <button type="button" className="record-btn stop" onClick={stopRecording} disabled={loading}>
                    Stop Recording
                  </button>
                )}

                {recording && <span className="record-status live">Recording…</span>}
                {recordedBlob && !recording && <span className="record-status">Recording ready ✔</span>}
              </div>
            </div>
          ) : (
            <label className="field">
              <span>Mel-spectrogram (.npy)</span>
              <input type="file" accept=".npy" onChange={(e) => setFile(e.target.files[0] || null)} />
            </label>
          )}

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Run Prediction"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <h2>Result: {result.label.toUpperCase()}</h2>
            <p><strong>Fusion Probability:</strong> {(result.fusion_prob * 100).toFixed(2)}%</p>
            <p><strong>Text Probability:</strong> {(result.text_prob * 100).toFixed(2)}%</p>
            <p><strong>Audio Probability:</strong> {(result.audio_prob * 100).toFixed(2)}%</p>
            {/* NEW CONTEXT RESULTS */}
            {result.sender_reputation !== undefined && (
              <p><strong>Sender Reputation:</strong> {(result.sender_reputation * 100).toFixed(1)}%</p>
            )}
            {result.url_risk !== undefined && (
              <p><strong>URL Risk:</strong> {(result.url_risk * 100).toFixed(1)}%</p>
            )}
            <p><strong>Threshold Used:</strong> {result.threshold.toFixed(4)}</p>
          </div>
        )}
      </div>
    </div>
  );
}
