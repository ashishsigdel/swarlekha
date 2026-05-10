import { useState, useCallback } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { AnimatePresence, motion } from "framer-motion";
import Navbar from "./components/Navbar";
import LoadingScreen from "./components/LoadingScreen";
import HomePage from "./pages/HomePage";
import CloningPage from "./pages/CloningPage";

function App() {
  const [ready, setReady] = useState(false);
  const onReady = useCallback(() => setReady(true), []);

  return (
    <BrowserRouter>
      <div className="flex flex-col" style={{ minHeight: "100vh", backgroundColor: "var(--bg)" }}>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: "var(--surface)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              fontSize: "13px",
            },
          }}
        />

        <AnimatePresence>
          {!ready && <LoadingScreen onReady={onReady} />}
        </AnimatePresence>

        {ready && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col flex-1"
          >
            <Navbar />
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/cloning" element={<CloningPage />} />
            </Routes>
          </motion.div>
        )}
      </div>
    </BrowserRouter>
  );
}

export default App;
