import { useEffect, useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

const STUDENT_ID = "student_" + Math.random().toString(36).slice(2, 8);

export default function App() {
  const [courseId, setCourseId] = useState("");
  const [lectureId, setLectureId] = useState("");
  const [videoUrl, setVideoUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [lectures, setLectures] = useState([]);
  const [selectedLectureId, setSelectedLectureId] = useState("");

  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const playerRef = useRef(null);

  const fetchLectures = async () => {
    try {
      const params = {};
      if (courseId.trim()) {
        params.course_id = courseId.trim();
      }
      const res = await axios.get(`${API}/lectures`, { params });
      const items = res.data?.lectures || [];
      setLectures(items);
      if (!selectedLectureId && items.length > 0) {
        setSelectedLectureId(items[0].lecture_id);
      }
    } catch {
      setLectures([]);
    }
  };

  useEffect(() => {
    fetchLectures();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress("Uploading...");
    const formData = new FormData();
    formData.append("file", file);
    if (courseId.trim()) {
      formData.append("course_id", courseId.trim());
    }

    try {
      setUploadProgress("Processing video — transcribing + extracting frames (this takes a few minutes)...");
      const res = await axios.post(`${API}/ingest`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.data.course_id) {
        setCourseId(res.data.course_id);
      }
      setLectureId(res.data.lecture_id);
      setVideoUrl(`${API}${res.data.video_url}`);
      const timings = res.data.stats?.timings || {};
      const totalSec = timings.total_seconds ? ` Total: ${timings.total_seconds}s.` : "";
      const llmCalls = res.data.stats?.visual_llm_calls ?? 0;
      const visualFrames = res.data.stats?.visual_frames ?? 0;
      setUploadProgress(
        `✅ Ready! ${res.data.stats.transcript_segments} transcript segments, ${res.data.stats.keyframes} keyframes, ${llmCalls}/${visualFrames} visual VLM calls.${totalSec}`
      );
      setMessages([{
        role: "system",
        text: `Lecture "${file.name}" loaded. Ask me anything about it!`
      }]);
      fetchLectures();
    } catch (err) {
      setUploadProgress("❌ Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
    }
  };

  const seekTo = (seconds) => {
    if (playerRef.current) {
      playerRef.current.currentTime = seconds;
      const playPromise = playerRef.current.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {});
      }
    }
  };

  const handleQuery = async () => {
    if (!inputQuery.trim() || !lectureId) return;

    const userMsg = { role: "user", text: inputQuery };
    setMessages((m) => [...m, userMsg]);
    setInputQuery("");
    setLoading(true);

    try {
      const res = await axios.post(`${API}/query`, {
        query: inputQuery,
        course_id: courseId || null,
        lecture_id: lectureId,
        student_id: STUDENT_ID,
      });

      const assistantMsg = {
        role: "assistant",
        text: res.data.answer,
        clips: res.data.clips,
        intent: res.data.intent,
        rewritten: res.data.rewritten_query !== inputQuery ? res.data.rewritten_query : null,
      };
      setMessages((m) => [...m, assistantMsg]);

      // Auto-seek to the top clip
      if (res.data.clips?.length > 0) {
        seekTo(res.data.clips[0].t_start);
      }
    } catch (err) {
      setMessages((m) => [...m, {
        role: "assistant",
        text: "Error: " + (err.response?.data?.detail || err.message),
        clips: []
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    if (!lectureId) return;

    try {
      await axios.post(`${API}/session/clear`, { student_id: STUDENT_ID });
    } catch {
      // Even if backend clear fails, we still clear local chat for UX.
    }

    const systemMsg = messages.find((m) => m.role === "system");
    if (systemMsg) {
      setMessages([systemMsg]);
    } else {
      setMessages([{ role: "system", text: "Chat cleared. You can continue asking about this lecture." }]);
    }
  };

  const handleLoadLecture = async () => {
    const chosen = lectures.find((l) => l.lecture_id === selectedLectureId);
    if (!chosen) return;

    setLectureId(chosen.lecture_id);
    setCourseId(chosen.course_id || "");
    setVideoUrl(`${API}${chosen.video_url}`);
    setUploadProgress(`Loaded existing lecture: ${chosen.file_name}`);

    try {
      await axios.post(`${API}/session/clear`, { student_id: STUDENT_ID });
    } catch {
      // Ignore if session clear fails.
    }

    setMessages([{ role: "system", text: `Lecture "${chosen.file_name}" loaded from library. Ask me anything about it!` }]);
  };

  const handleClearLibrary = async () => {
    const ok = window.confirm("Clear all saved lectures, frames, and indexes? This cannot be undone.");
    if (!ok) return;

    try {
      await axios.post(`${API}/lectures/clear`, { remove_graphs: true });
      setLectures([]);
      setSelectedLectureId("");
      setLectureId("");
      setVideoUrl(null);
      setMessages([{ role: "system", text: "Lecture library cleared." }]);
      setUploadProgress("✅ Cleared saved lectures and indexes.");
      await fetchLectures();
    } catch (err) {
      setUploadProgress("❌ Clear failed: " + (err.response?.data?.detail || err.message));
    }
  };

  const fmt = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎓 LectureRAG</h1>
        <div className="course-input">
          <label>Course ID (optional):</label>
          <input value={courseId} onChange={(e) => setCourseId(e.target.value)} placeholder="auto: general" />
          <button type="button" onClick={handleClearLibrary}>Clear Library</button>
        </div>
      </header>

      <div className="main-layout">
        <div className="video-panel">
          {!videoUrl ? (
            <div className="upload-area">
              {lectures.length > 0 && (
                <div className="existing-lectures">
                  <label>Load Existing Lecture</label>
                  <div className="existing-lectures-row">
                    <select value={selectedLectureId} onChange={(e) => setSelectedLectureId(e.target.value)}>
                      {lectures.map((l) => (
                        <option key={l.lecture_id} value={l.lecture_id}>
                          {l.file_name} ({l.lecture_id})
                        </option>
                      ))}
                    </select>
                    <button type="button" onClick={handleLoadLecture}>Load</button>
                    <button type="button" onClick={handleClearLibrary}>Clear Library</button>
                  </div>
                </div>
              )}
              <label className="upload-btn">
                {uploading ? "⏳ Processing..." : "📁 Upload Lecture Video"}
                <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} hidden />
              </label>
              {uploadProgress && <p className="progress-text">{uploadProgress}</p>}
            </div>
          ) : (
            <>
              <video
                ref={playerRef}
                src={videoUrl}
                controls
                preload="metadata"
                className="lecture-video"
              />
              {uploadProgress && <p className="progress-text">{uploadProgress}</p>}
            </>
          )}
        </div>

        <div className="chat-panel">
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                {msg.rewritten && (
                  <div className="rewrite-badge">
                    🔄 Interpreted as: "{msg.rewritten}"
                  </div>
                )}
                {msg.intent && (
                  <span className={`intent-badge intent-${msg.intent}`}>
                    {msg.intent}
                  </span>
                )}
                <p>{msg.text}</p>
                {msg.clips?.map((clip, j) => (
                  <div key={j} className={`clip-card confidence-${clip.confidence?.toLowerCase() || 'low'}`}>
                    <button
                      className="timestamp-btn"
                      onClick={() => seekTo(clip.t_start)}
                      title="Click to jump to this moment"
                    >
                      ▶ {fmt(clip.t_start)} → {fmt(clip.t_end)}
                    </button>
                    <span className={`confidence-tag ${clip.confidence?.toLowerCase() || 'low'}`}>
                      {clip.confidence || 'LOW'}
                    </span>
                    {clip.content_type && clip.content_type !== "talking_head" && (
                      <span className="content-type-tag">{clip.content_type}</span>
                    )}
                    {clip.see_also?.length > 0 && (
                      <div className="see-also">
                        <span>See also: </span>
                        {clip.see_also.map((sa, k) => (
                          <button key={k} className="see-also-btn"
                            onClick={() => seekTo(sa.t_start)}>
                            {sa.concept} ({fmt(sa.t_start)})
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ))}
            {loading && <div className="message assistant loading">⏳ Thinking...</div>}
          </div>

          <div className="input-area">
            <input
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleQuery()}
              placeholder={lectureId ? "Ask about the lecture..." : "Upload a video first"}
              disabled={!lectureId || loading}
            />
            <button onClick={handleQuery} disabled={!lectureId || loading}>
               Send
            </button>
            <button className="clear-btn" onClick={handleClearChat} disabled={!lectureId || loading}>
              Clear Chat
            </button>
          </div>

          {lectureId && messages.length <= 1 && (
            <div className="suggestions">
              {[
                "Summarize the key takeaways",
                "Show me the most important diagrams",
                "Quiz me on this lecture",
                "What was discussed in the beginning?"
              ].map((q) => (
                <button key={q} className="suggestion-pill"
                  onClick={() => { setInputQuery(q); }}>
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
