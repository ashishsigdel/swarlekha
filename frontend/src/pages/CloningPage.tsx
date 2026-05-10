import VoiceGenerator from "../components/VoiceGenerator";

const CloningPage = () => {
  return (
    <div
      className="flex flex-col overflow-hidden"
      style={{ height: "calc(100vh - 57px)", backgroundColor: "var(--bg)" }}
    >
      <VoiceGenerator />
    </div>
  );
};

export default CloningPage;
