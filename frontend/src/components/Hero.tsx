import { Link } from "react-router-dom";
import { Mic2, Globe, Zap, ArrowRight } from "lucide-react";

const features = [
  {
    icon: Mic2,
    title: "Voice Cloning",
    description:
      "Clone any voice with just a short audio sample using state-of-the-art neural models.",
  },
  {
    icon: Globe,
    title: "Multilingual",
    description:
      "Generate natural speech in both English and Nepali with dedicated language models.",
  },
  {
    icon: Zap,
    title: "Fast Inference",
    description:
      "Optimized for speed — get high-quality audio output in seconds on modern hardware.",
  },
];

const Hero = () => {
  return (
    <div
      className="flex flex-col"
      style={{ minHeight: "calc(100vh - 57px)" }}
    >
      {/* Hero */}
      <section className="flex-1 flex items-center justify-center px-6 py-24">
        <div className="max-w-3xl mx-auto text-center">
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8 badge badge-primary"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
            English & Nepali voice synthesis
          </div>

          <h1
            className="text-5xl md:text-6xl font-bold tracking-tight mb-6"
            style={{ color: "var(--text)", lineHeight: 1.1 }}
          >
            Text to speech,{" "}
            <span style={{ color: "var(--primary)" }}>your voice</span>
          </h1>

          <p
            className="text-lg md:text-xl mb-10 max-w-2xl mx-auto leading-relaxed"
            style={{ color: "var(--text-muted)" }}
          >
            Swarlekha TTS turns any text into natural-sounding audio. Upload a
            reference clip to clone a voice, or use the built-in default voice
            — in English or Nepali.
          </p>

          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Link to="/cloning" className="btn-primary inline-flex items-center gap-2">
              Open Studio
              <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href="/api/health"
              target="_blank"
              rel="noreferrer"
              className="btn-secondary"
            >
              API Status
            </a>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 pb-24">
        <div className="max-w-4xl mx-auto">
          <div
            className="grid md:grid-cols-3 gap-4"
          >
            {features.map((f) => (
              <div key={f.title} className="card">
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center mb-4"
                  style={{ backgroundColor: "var(--primary-light)" }}
                >
                  <f.icon className="w-4.5 h-4.5" style={{ color: "var(--primary)" }} />
                </div>
                <h3
                  className="font-semibold mb-1.5 text-sm"
                  style={{ color: "var(--text)" }}
                >
                  {f.title}
                </h3>
                <p className="text-sm leading-relaxed" style={{ color: "var(--text-muted)" }}>
                  {f.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Hero;
