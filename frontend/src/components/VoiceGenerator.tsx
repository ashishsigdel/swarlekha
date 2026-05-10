import { useState, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Upload, Mic, Play, Download, X, Volume2, Square } from "lucide-react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { generateVoiceStream, cancelGeneration, getResultUrl, GenerationPhase } from "../services/api";

type Language = "english" | "nepali";

type ProgressState =
  | null
  | { phase: "preparing" }
  | { phase: "generating"; tokens: number }
  | { phase: "synthesizing" }
  | { phase: "complete" }
  | { phase: "cancelled" };

const PHASE_LABELS: Record<string, string> = {
  preparing: "Preparing voice…",
  generating: "Generating speech tokens",
  synthesizing: "Synthesizing audio…",
  complete: "Done",
  cancelled: "Cancelled",
};

const PHASES = ["preparing", "generating", "synthesizing"] as const;

function phaseIndex(phase: string): number {
  return PHASES.indexOf(phase as typeof PHASES[number]);
}

const VoiceGenerator = () => {
  const [text, setText] = useState("");
  const [voiceName, setVoiceName] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [progress, setProgress] = useState<ProgressState>(null);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [useClone, setUseClone] = useState(false);
  const [language, setLanguage] = useState<Language>("english");

  const sessionIdRef = useRef<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const isGenerating =
    progress !== null &&
    progress.phase !== "complete" &&
    progress.phase !== "cancelled";

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setAudioFile(accepted[0]);
      setUseClone(true);
      toast.success("Reference audio uploaded");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [".wav", ".mp3", ".m4a", ".ogg"] },
    maxFiles: 1,
  });

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        setAudioFile(new File([blob], "recorded.wav", { type: "audio/wav" }));
        setUseClone(true);
        stream.getTracks().forEach((t) => t.stop());
        toast.success("Recording saved");
      };
      recorder.start();
      setIsRecording(true);
      toast.success("Recording…");
    } catch {
      toast.error("Cannot access microphone");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleGenerate = async () => {
    if (!text.trim()) { toast.error("Enter some text first"); return; }
    setProgress({ phase: "preparing" });
    setGeneratedAudio(null);
    sessionIdRef.current = null;

    let audioUrl = "";

    try {
      await generateVoiceStream(
        text,
        useClone ? audioFile : null,
        voiceName || "generated",
        language,
        (event: GenerationPhase) => {
          if (event.phase === "session") {
            sessionIdRef.current = event.session_id;
          } else if (event.phase === "complete") {
            audioUrl = event.audio_url;
            setProgress({ phase: "complete" });
          } else if (event.phase === "cancelled") {
            setProgress({ phase: "cancelled" });
          } else if (event.phase === "error") {
            throw new Error(event.detail);
          } else {
            setProgress(event as ProgressState);
          }
        }
      );

      if (audioUrl) {
        setGeneratedAudio(getResultUrl(audioUrl));
        toast.success("Audio generated");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Generation failed";
      toast.error(msg);
      setProgress(null);
    }
  };

  const handleStop = async () => {
    if (!sessionIdRef.current) return;
    try {
      await cancelGeneration(sessionIdRef.current);
    } catch {
      // If the session already ended that's fine
    }
  };

  const handleDownload = () => {
    if (!generatedAudio) return;
    const a = document.createElement("a");
    a.href = generatedAudio;
    a.download = `${voiceName || "generated"}_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const tokenCount =
    progress?.phase === "generating"
      ? (progress as { phase: "generating"; tokens: number }).tokens
      : null;

  return (
    <div className="h-full flex flex-col overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
      {/* Toolbar */}
      <div
        className="flex-none flex items-center justify-between px-5 py-3 border-b gap-4"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="flex items-center gap-1 p-1 rounded-lg border"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-subtle)" }}
        >
          {(["english", "nepali"] as Language[]).map((lang) => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              disabled={isGenerating}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
                language === lang ? "tab-active" : "tab-inactive"
              }`}
            >
              {lang === "english" ? "🇺🇸 English" : "🇳🇵 Nepali"}
            </button>
          ))}
        </div>

        <div
          className="flex items-center gap-1 p-1 rounded-lg border"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-subtle)" }}
        >
          <button
            onClick={() => { setUseClone(false); setAudioFile(null); }}
            disabled={isGenerating}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
              !useClone ? "tab-active" : "tab-inactive"
            }`}
          >
            Default Voice
          </button>
          <button
            onClick={() => setUseClone(true)}
            disabled={isGenerating}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
              useClone ? "tab-active" : "tab-inactive"
            }`}
          >
            Clone Voice
          </button>
        </div>

        <input
          type="text"
          value={voiceName}
          onChange={(e) => setVoiceName(e.target.value)}
          disabled={isGenerating}
          placeholder="Session name…"
          className="input-field w-40 text-xs py-2"
        />
      </div>

      {/* Main workspace */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel */}
        <div
          className="w-[48%] flex flex-col border-r min-h-0 overflow-hidden"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex-1 flex flex-col p-4 min-h-0">
            <label className="label">Text</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={isGenerating}
              placeholder={
                language === "nepali"
                  ? "यहाँ आफ्नो पाठ लेख्नुहोस्…"
                  : "Enter the text you want to synthesize…"
              }
              className="input-field flex-1 resize-none min-h-0 text-sm leading-relaxed"
              maxLength={5000}
              style={{
                fontFamily: language === "nepali" ? "'Noto Sans Devanagari', sans-serif" : undefined,
              }}
            />
            <div className="mt-2 text-xs text-right" style={{ color: "var(--text-subtle)" }}>
              {text.length} / 5000
            </div>
          </div>

          <AnimatePresence>
            {useClone && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden border-t"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="p-4">
                  <label className="label">Reference Audio</label>
                  <div
                    {...getRootProps()}
                    className="rounded-lg border-2 border-dashed px-4 py-5 text-center cursor-pointer transition-all duration-150"
                    style={{
                      borderColor: isDragActive ? "var(--primary)" : "var(--border-strong)",
                      backgroundColor: isDragActive ? "var(--primary-light)" : "var(--bg-subtle)",
                    }}
                  >
                    <input {...getInputProps()} />
                    {audioFile ? (
                      <div className="flex items-center justify-center gap-2 text-sm">
                        <Volume2 className="w-4 h-4" style={{ color: "var(--primary)" }} />
                        <span style={{ color: "var(--text)" }}>{audioFile.name}</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); setAudioFile(null); }}
                          style={{ color: "var(--text-subtle)" }}
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <Upload className="w-5 h-5 mx-auto mb-2" style={{ color: "var(--text-subtle)" }} />
                        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                          Drop audio here or click to browse
                        </p>
                        <p className="text-xs mt-1" style={{ color: "var(--text-subtle)" }}>
                          WAV, MP3, M4A, OGG
                        </p>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-3 mt-3">
                    <div className="flex-1 border-t" style={{ borderColor: "var(--border)" }} />
                    <span className="text-xs" style={{ color: "var(--text-subtle)" }}>or</span>
                    <div className="flex-1 border-t" style={{ borderColor: "var(--border)" }} />
                  </div>

                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isGenerating}
                    className="mt-3 w-full btn-secondary flex items-center justify-center gap-2 text-sm"
                    style={isRecording ? { color: "#ef4444", borderColor: "#ef4444" } : {}}
                  >
                    {isRecording ? (
                      <><Square className="w-3.5 h-3.5 fill-current" /> Stop Recording</>
                    ) : (
                      <><Mic className="w-3.5 h-3.5" /> Record Voice</>
                    )}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Generate / Progress / Stop */}
          <div className="p-4 border-t" style={{ borderColor: "var(--border)" }}>
            {isGenerating ? (
              <div className="space-y-3">
                <ProgressBar progress={progress} tokenCount={tokenCount} />
                <button
                  onClick={handleStop}
                  className="w-full btn-secondary flex items-center justify-center gap-2 text-sm"
                  style={{ color: "#ef4444", borderColor: "#ef4444" }}
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  Stop Generation
                </button>
              </div>
            ) : (
              <button
                onClick={handleGenerate}
                disabled={!text.trim()}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4 fill-current" />
                Generate
              </button>
            )}
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 min-h-0">
          <AnimatePresence mode="wait">
            {generatedAudio ? (
              <motion.div
                key="audio"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="w-full max-w-sm space-y-4"
              >
                <div className="card">
                  <div className="flex items-center gap-2 mb-4">
                    <Volume2 className="w-4 h-4" style={{ color: "var(--primary)" }} />
                    <span className="text-sm font-medium" style={{ color: "var(--text)" }}>
                      Generated Audio
                    </span>
                    <span className="ml-auto badge badge-primary">{language}</span>
                  </div>
                  <div
                    className="flex items-end justify-center gap-0.5 h-12 rounded-lg px-3 mb-4"
                    style={{ backgroundColor: "var(--bg-subtle)" }}
                  >
                    {[...Array(28)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="rounded-full flex-1"
                        style={{ backgroundColor: "var(--primary)", minWidth: 2 }}
                        animate={{ height: ["30%", `${40 + Math.sin(i * 0.8) * 50}%`, "30%"] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.04 }}
                      />
                    ))}
                  </div>
                  <audio ref={audioRef} src={generatedAudio} className="w-full" controls />
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => audioRef.current?.play()}
                    className="flex-1 btn-secondary flex items-center justify-center gap-2 text-sm"
                  >
                    <Play className="w-3.5 h-3.5" /> Play
                  </button>
                  <button
                    onClick={handleDownload}
                    className="flex-1 btn-primary flex items-center justify-center gap-2 text-sm"
                  >
                    <Download className="w-3.5 h-3.5" /> Download
                  </button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center"
              >
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                  style={{ backgroundColor: "var(--bg-muted)" }}
                >
                  <Volume2 className="w-7 h-7" style={{ color: "var(--text-subtle)" }} />
                </div>
                <p className="text-sm font-medium mb-1" style={{ color: "var(--text-muted)" }}>
                  No audio yet
                </p>
                <p className="text-xs" style={{ color: "var(--text-subtle)" }}>
                  Enter text and click Generate
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

function ProgressBar({
  progress,
  tokenCount,
}: {
  progress: ProgressState;
  tokenCount: number | null;
}) {
  if (!progress) return null;
  const phase = progress.phase;
  const pIdx = phaseIndex(phase);
  const label = PHASE_LABELS[phase] ?? phase;

  return (
    <div className="space-y-2">
      {/* Phase step dots */}
      <div className="flex items-center gap-1.5">
        {PHASES.map((p, i) => {
          const done = pIdx > i;
          const active = pIdx === i;
          return (
            <div key={p} className="flex items-center gap-1.5 flex-1">
              <motion.div
                className="w-2 h-2 rounded-full flex-shrink-0"
                animate={{
                  backgroundColor:
                    done || active ? "var(--primary)" : "var(--border-strong)",
                  opacity: done || active ? 1 : 0.35,
                  scale: active ? [1, 1.3, 1] : 1,
                }}
                transition={active ? { duration: 1, repeat: Infinity } : { duration: 0.3 }}
              />
              {i < PHASES.length - 1 && (
                <div
                  className="flex-1 h-px transition-all duration-500"
                  style={{ backgroundColor: done ? "var(--primary)" : "var(--border)" }}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Label + token count */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
          {label}
        </span>
        {tokenCount !== null && (
          <span className="text-xs tabular-nums" style={{ color: "var(--text-subtle)" }}>
            {tokenCount} tokens
          </span>
        )}
      </div>

      {/* Track */}
      <div className="h-1 rounded-full overflow-hidden" style={{ backgroundColor: "var(--bg-muted)" }}>
        {phase === "generating" ? (
          <motion.div
            className="h-full w-1/3 rounded-full"
            style={{ backgroundColor: "var(--primary)" }}
            animate={{ x: ["-100%", "300%"] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : (
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundColor: phase === "cancelled" ? "#ef4444" : "var(--primary)" }}
            animate={{
              width:
                phase === "preparing"
                  ? "20%"
                  : phase === "synthesizing"
                  ? "75%"
                  : phase === "complete" || phase === "cancelled"
                  ? "100%"
                  : "0%",
            }}
            transition={{ duration: 0.4 }}
          />
        )}
      </div>
    </div>
  );
}

export default VoiceGenerator;
