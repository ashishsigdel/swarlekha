import { Link, useLocation } from "react-router-dom";
import { Moon, Sun, Mic2 } from "lucide-react";
import { useTheme } from "../hooks/useTheme";

const Navbar = () => {
  const { dark, toggle } = useTheme();
  const { pathname } = useLocation();

  return (
    <header
      className="sticky top-0 z-50 border-b"
      style={{
        borderColor: "var(--border)",
        backgroundColor: "var(--bg)",
      }}
    >
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link
          to="/"
          className="flex items-center gap-2 font-semibold text-sm"
          style={{ color: "var(--text)" }}
        >
          <Mic2 className="w-4 h-4" style={{ color: "var(--primary)" }} />
          Swarlekha
        </Link>

        <nav className="flex items-center gap-1">
          <Link
            to="/"
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-150 ${
              pathname === "/" ? "tab-active" : "tab-inactive"
            }`}
          >
            Home
          </Link>
          <Link
            to="/cloning"
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-150 ${
              pathname === "/cloning" ? "tab-active" : "tab-inactive"
            }`}
          >
            Studio
          </Link>
        </nav>

        <button
          onClick={toggle}
          className="btn-ghost w-9 h-9 flex items-center justify-center rounded-lg p-0"
          aria-label="Toggle theme"
        >
          {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  );
};

export default Navbar;
