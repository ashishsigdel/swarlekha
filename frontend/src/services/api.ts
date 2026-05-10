import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: API_BASE_URL });

export type ModelLoadStatus = "pending" | "loading" | "loaded" | "error";

export interface HealthResponse {
  status: string;
  device: string;
  model_status: { english: ModelLoadStatus; nepali: ModelLoadStatus };
  english_model_loaded: boolean;
  nepali_model_loaded: boolean;
}

export type GenerationPhase =
  | { phase: "session"; session_id: string }
  | { phase: "preparing" }
  | { phase: "generating"; tokens: number }
  | { phase: "synthesizing" }
  | { phase: "complete"; audio_url: string }
  | { phase: "cancelled" }
  | { phase: "error"; detail: string };

export const healthCheck = async (): Promise<HealthResponse> => {
  const res = await api.get<HealthResponse>("/api/health");
  return res.data;
};

/**
 * Stream voice generation via SSE.
 * `onEvent` is called for each progress event.
 * Returns a Promise that resolves on complete/cancelled, rejects on error.
 */
export const generateVoiceStream = (
  text: string,
  referenceAudio: File | null,
  voiceName: string,
  language: "english" | "nepali",
  onEvent: (event: GenerationPhase) => void
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("text", text);
    formData.append("voice_name", voiceName);
    formData.append("language", language);
    if (referenceAudio) formData.append("reference_audio", referenceAudio);

    fetch(`${API_BASE_URL}/api/generate-stream`, {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((e) => { throw new Error(e.detail || "Generation failed"); });
        }

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const pump = (): Promise<void> =>
          reader.read().then(({ done, value }) => {
            if (done) { resolve(); return; }
            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split("\n\n");
            buffer = chunks.pop() ?? "";

            for (const chunk of chunks) {
              const line = chunk.trim();
              if (!line.startsWith("data:")) continue;
              try {
                const event: GenerationPhase = JSON.parse(line.slice(5).trim());
                onEvent(event);
                if (event.phase === "complete" || event.phase === "cancelled") {
                  resolve(); return;
                }
                if (event.phase === "error") {
                  reject(new Error(event.detail)); return;
                }
              } catch { /* malformed chunk */ }
            }
            return pump();
          });

        return pump();
      })
      .catch(reject);
  });
};

export const cancelGeneration = async (sessionId: string): Promise<void> => {
  await api.post(`/api/cancel/${sessionId}`);
};

export const getResultUrl = (audioUrl: string): string =>
  `${API_BASE_URL}${audioUrl}`;

export default api;
