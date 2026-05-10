import { Link } from "react-router-dom";
import { Mic2 } from "lucide-react";

const Footer = () => {
  return (
    <footer
      className="border-t px-6 py-8"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <Link
          to="/"
          className="flex items-center gap-2 text-sm font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          <Mic2 className="w-4 h-4" style={{ color: "var(--primary)" }} />
          Swarlekha TTS
        </Link>

        <div className="flex items-center gap-5 text-xs" style={{ color: "var(--text-subtle)" }}>
          <Link to="/cloning" className="nav-link">Studio</Link>
          <a href="/api/health" className="nav-link" target="_blank" rel="noreferrer">
            API
          </a>
          <span>© {new Date().getFullYear()} Swarlekha</span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
