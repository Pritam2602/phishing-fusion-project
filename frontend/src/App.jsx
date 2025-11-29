import React, { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";

function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [useRecording, setUseRecording] = useState(true);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    try {
      setError("");
      setResult(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        // stop all tracks so the mic is released
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setRecording(true);
      setRecordedBlob(null);
    } catch (err) {
      setError("Could not access microphone. Please allow mic permissions.");
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecording(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    const hasText = !!text.trim();
    const hasAudio = useRecording ? !!recordedBlob : !!file;

    if (!hasText && !hasAudio) {
      setError("Please provide text and/or audio.");
      return;
    }

    const formData = new FormData();
    formData.append("text", hasText ? text : "");

    let url = "";

    if (useRecording && recordedBlob) {
      formData.append("audio_wav", recordedBlob, "recording.webm");
      url = `${API_BASE}/predict_recorded`;
    } else if (!useRecording && file) {
      formData.append("audio_file", file);
      url = `${API_BASE}/predict`;
    } else if (hasText) {
      // Text-only prediction (no audio attached)
      url = `${API_BASE}/predict`;
    } else {
      setError("Please provide text and/or audio.");
      return;
    }

    try {
      setLoading(true);
      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "Request failed");
      }

      const data = await res.json();
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
          Enter email/voice text and optionally add audio (recording or .npy) to check if it is
          phishing. The model works with text only, audio only, or both.
        </p>

        <form onSubmit={handleSubmit} className="form">
          <label className="field">
            <span>Text message</span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              placeholder="Your account has been compromised..."
            />
          </label>

          <div className="field">
            <span>Audio source</span>
            <div className="toggle-row">
              <button
                type="button"
                className={useRecording ? "toggle active" : "toggle"}
                onClick={() => setUseRecording(true)}
              >
                Record with microphone
              </button>
              <button
                type="button"
                className={!useRecording ? "toggle active" : "toggle"}
                onClick={() => setUseRecording(false)}
              >
                Upload .npy file
              </button>
            </div>
          </div>

          {useRecording ? (
            <div className="field">
              <span>Recorded audio</span>
              <div className="record-row">
                {!recording ? (
                  <button
                    type="button"
                    onClick={startRecording}
                    disabled={loading}
                    className="record-btn"
                  >
                    {recordedBlob ? "Re-record" : "Start recording"}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopRecording}
                    disabled={loading}
                    className="record-btn stop"
                  >
                    Stop recording
                  </button>
                )}
                {recordedBlob && !recording && (
                  <span className="record-status">Recording ready ✔</span>
                )}
                {recording && <span className="record-status live">Recording…</span>}
              </div>
            </div>
          ) : (
            <label className="field">
              <span>Mel-spectrogram (.npy)</span>
              <input
                type="file"
                accept=".npy"
                onChange={(e) => setFile(e.target.files[0] || null)}
              />
            </label>
          )}

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Run prediction"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <h2>Result: {result.label.toUpperCase()}</h2>
            <p>
              <strong>Fusion probability:</strong> {(result.fusion_prob * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Text probability:</strong> {(result.text_prob * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Audio probability:</strong> {(result.audio_prob * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Threshold used:</strong> {result.threshold.toFixed(4)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;


