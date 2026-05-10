import Hero from "../components/Hero";
import Footer from "../components/Footer";

const HomePage = () => {
  return (
    <div className="flex flex-col min-h-screen" style={{ backgroundColor: "var(--bg)" }}>
      <Hero />
      <Footer />
    </div>
  );
};

export default HomePage;
