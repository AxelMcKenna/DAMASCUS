import Hero from './components/Hero'
import Thesis from './components/Thesis'
import Compartments from './components/Compartments'
import Anatomy from './components/Anatomy'
import DataFlow from './components/DataFlow'
import Milestones from './components/Milestones'
import ModelSpecs from './components/ModelSpecs'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="min-h-screen overflow-x-hidden">
      <main>
        <Hero />
        <Thesis />
        <Compartments />
        <Anatomy />
        <DataFlow />
        <Milestones />
        <ModelSpecs />
      </main>
      <Footer />
      <div className="vignette" />
      <div className="grain" />
    </div>
  )
}
