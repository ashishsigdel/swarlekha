import { useEffect, useState } from "react";
import { Mic2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { healthCheck, ModelLoadStatus } from "../services/api";

interface ModelState {
  english: ModelLoadStatus;
  nepali: ModelLoadStatus;
}

interface Props {
  onReady: () => void;
}

const STATUS_LABEL: Record<ModelLoadStatus, string> = {
  pending: "Waiting…",
  loading: "Loading…",
  loaded: "Ready",
  error: "Failed",
};

const STATUS_COLOR: Record<ModelLoadStatus, string> = {
  pending: "var(--text-subtle)",
  loading: "var(--primary)",
  loaded: "#22c55e",
  error: "#ef4444",
};

export default function LoadingScreen({ onReady }: Props) {
  const [models, setModels] = useState<ModelState>({ english: "pending", nepali: "pending" });
  const [backendDown, setBackendDown] = useState(false);

  useEffect(() => {
    let rafId: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const health = await healthCheck();
        const ms = health.model_status ?? {
          english: health.english_model_loaded ? "loaded" : "pending",
          nepali: health.nepali_model_loaded ? "loaded" : "pending",
        };
        setModels(ms as ModelState);
        setBackendDown(false);

        const bothDone =
          (ms.english === "loaded" || ms.english === "error") &&
          (ms.nepali === "loaded" || ms.nepali === "error");

        if (bothDone) {
          // Small delay so the user sees the "Ready" state before the screen disappears
          setTimeout(onReady, 600);
        } else {
          rafId = setTimeout(poll, 1200);
        }
      } catch {
        setBackendDown(true);
        rafId = setTimeout(poll, 2000);
      }
    };

    poll();
    return () => clearTimeout(rafId);
  }, [onReady]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-10"
        style={{ backgroundColor: "var(--bg)" }}
      >
        {/* Logo */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col items-center gap-3"
        >
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center"
            style={{ backgroundColor: "var(--primary-light)" }}
          >
            <Mic2 className="w-7 h-7" style={{ color: "var(--primary)" }} />
          </div>
          <span className="text-lg font-semibold" style={{ color: "var(--text)" }}>
            Swarlekha TTS
          </span>
        </motion.div>

        {/* Model status cards */}
        <div className="flex flex-col gap-3 w-72">
          {(["english", "nepali"] as const).map((lang) => {
            const status = models[lang];
            return (
              <motion.div
                key={lang}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="card flex items-center justify-between"
              >
                <div className="flex items-center gap-3">
                  <span className="text-base">{lang === "english" ? "🇺🇸" : "🇳🇵"}</span>
                  <span className="text-sm font-medium capitalize" style={{ color: "var(--text)" }}>
                    {lang} model
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {status === "loading" && (
                    <motion.div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: "var(--primary)" }}
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                  )}
                  <span
                    className="text-xs font-medium"
                    style={{ color: STATUS_COLOR[status] }}
                  >
                    {STATUS_LABEL[status]}
                  </span>
                </div>
              </motion.div>
            );
          })}

          {backendDown && (
            <p className="text-xs text-center" style={{ color: "var(--text-subtle)" }}>
              Waiting for backend to start…
            </p>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
