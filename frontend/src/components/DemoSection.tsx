import { motion } from "framer-motion";
import { Volume2 } from "lucide-react";

interface DemoItem {
  id: string;
  title: string;
  description: string;
  audioUrl: string;
  type: "default" | "cloned";
}

const demoData: DemoItem[] = [
  {
    id: "1",
    title: "Ashish's Voice",
    description: "Original voice",
    audioUrl: "/demo/ashish/default_voice.wav",
    type: "default",
  },
  {
    id: "2",
    title: "Cloned Voice - Sample 1",
    description: "Cloned Voice",
    audioUrl: "/demo/ashish/cloned_voice_1.wav",
    type: "cloned",
  },
  {
    id: "3",
    title: "Cloned Voice - Sample 2",
    description: "Another example of high-quality voice cloning",
    audioUrl: "/demo/ashish/cloned_voice_2.wav",
    type: "cloned",
  },
  {
    id: "4",
    title: "Indra's Voice",
    description: "Original voice",
    audioUrl: "/demo/indra/default_voice.wav",
    type: "default",
  },
  {
    id: "5",
    title: "Cloned Voice - Sample 1",
    description: "Cloned Voice",
    audioUrl: "/demo/indra/cloned_voice_1.wav",
    type: "cloned",
  },
  {
    id: "6",
    title: "Cloned Voice - Sample 2",
    description: "Another example of high-quality voice cloning",
    audioUrl: "/demo/indra/cloned_voice_2.wav",
    type: "cloned",
  },
];

const DemoSection = () => {
  return (
    <section className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 gradient-text">
            Demo Examples
          </h2>
          <p className="text-center text-gray-400 mb-12 max-w-2xl mx-auto">
            Listen to examples of our voice generation and cloning capabilities
          </p>

          <div className="grid md:grid-cols-3 gap-6">
            {demoData.map((demo, index) => (
              <motion.div
                key={demo.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="card group hover:scale-105 transition-transform duration-300"
              >
                <div className="flex items-center justify-between mb-4">
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      demo.type === "default"
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-purple-500/20 text-purple-400"
                    }`}
                  >
                    {demo.type === "default" ? "Default" : "Cloned"}
                  </div>
                  <Volume2 className="w-5 h-5 text-gray-400" />
                </div>

                <h3 className="text-xl font-semibold mb-2">{demo.title}</h3>
                <p className="text-gray-400 text-sm mb-4">{demo.description}</p>

                <div className="space-y-3">
                  <audio src={demo.audioUrl} controls className="w-full" />
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-12 text-center">
            <p className="text-gray-400 mb-4">
              Note: These are demo placeholders. Replace with your actual
              generated audio files.
            </p>
            <motion.div whileHover={{ scale: 1.05 }} className="inline-block">
              <a
                href="#generator"
                className="btn-primary inline-flex items-center gap-2"
              >
                Try It Yourself
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 5l7 7m0 0l-7 7m7-7H3"
                  />
                </svg>
              </a>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default DemoSection;
