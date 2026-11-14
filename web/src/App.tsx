import Hero from './components/Hero'
import Thesis from './components/Thesis'
import Compartments from './components/Compartments'
import Anatomy from './components/Anatomy'
import DataFlow from './components/DataFlow'
import Milestones from './components/Milestones'
import ModelSpecs from './components/ModelSpecs'
import Footer from './components/Footer'
import Creator from './pages/Creator'
import TechnicalArticle from './pages/TechnicalArticle'

function Page({ path }: { path: string }) {
  if (path === '/axel-mckenna') return <Creator />
  if (path === '/writing/building-damascus') return <TechnicalArticle />
  if (path !== '/') {
    return (
      <section className="mx-auto max-w-4xl px-6 py-24">
        <h1 className="text-4xl">Page not found</h1>
        <a href="/" className="mt-8 inline-block text-forge-300 underline">Return to Damascus</a>
      </section>
    )
  }
  return (
    <>
      <Hero />
      <Thesis />
      <Compartments />
      <Anatomy />
      <DataFlow />
      <Milestones />
      <ModelSpecs />
    </>
  )
}

export default function App({ path = '/' }: { path?: string }) {
  return (
    <div className="min-h-screen overflow-x-hidden">
      <main>
        <Page path={path.replace(/\/$/, '') || '/'} />
      </main>
      <Footer />
      <div className="vignette" />
      <div className="grain" />
    </div>
  )
}
