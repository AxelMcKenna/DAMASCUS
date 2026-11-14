export const SITE_URL = 'https://damascus-llm.dev'
export const CREATOR_URL = `${SITE_URL}/axel-mckenna`
const personId = `${CREATOR_URL}#person`
const softwareId = `${SITE_URL}/#software`

export const pages = [
  {
    path: '/',
    title: 'DAMASCUS — Apple Silicon LLM Inference Engine by Axel McKenna',
    description: 'Axel McKenna is building DAMASCUS, a from-scratch LLM inference engine for Apple M1 in C++20 and Metal. Explore its architecture and verified CPU results.',
    type: 'website',
  },
  {
    path: '/axel-mckenna',
    title: 'Axel McKenna — Creator of DAMASCUS and Liquorfy',
    description: 'Meet Axel McKenna, creator of DAMASCUS and Liquorfy. Explore his work on Apple Silicon inference, C++20, Metal, and New Zealand price comparison.',
    type: 'profile',
  },
  {
    path: '/writing/building-damascus',
    title: 'Building DAMASCUS: LLM Inference on Apple M1 — Axel McKenna',
    description: 'Axel McKenna explains DAMASCUS: GGUF loading, transformer inference, same-weights CPU validation, and a reproducible benchmark plan for Metal and AMX.',
    type: 'article',
  },
] as const

export function getPage(path: string) {
  const normalized = path.replace(/\/$/, '') || '/'
  return pages.find(page => page.path === normalized) ?? {
    path: '/404', title: 'Page not found — DAMASCUS',
    description: 'This DAMASCUS page could not be found.', type: 'website',
  }
}

export function structuredData(path: string) {
  const page = getPage(path)
  const url = `${SITE_URL}${page.path}`
  const person = {
    '@type': 'Person', '@id': personId, name: 'Axel McKenna', url: CREATOR_URL,
    sameAs: ['https://github.com/AxelMcKenna'],
    description: 'Creator of DAMASCUS and Liquorfy.',
  }
  const software = {
    '@type': 'SoftwareSourceCode', '@id': softwareId, name: 'DAMASCUS',
    url: `${SITE_URL}/`, codeRepository: 'https://github.com/AxelMcKenna/DAMASCUS',
    description: 'A from-scratch LLM inference engine for Apple M1, written in C++20 with a planned Metal and AMX compute path.',
    programmingLanguage: ['C++', 'Objective-C++', 'Metal'],
    runtimePlatform: 'Apple Silicon M1', author: { '@id': personId },
  }
  const entity = page.type === 'profile' ? {
    '@type': 'ProfilePage', '@id': url, url, name: page.title,
    mainEntity: { '@id': personId },
  } : page.type === 'article' ? {
    '@type': 'TechArticle', '@id': url, url, headline: 'Building DAMASCUS: a language model from file to forward pass',
    description: page.description, author: { '@id': personId, '@type': 'Person', name: 'Axel McKenna', url: CREATOR_URL },
    mainEntityOfPage: url, about: { '@id': softwareId }, inLanguage: 'en',
  } : {
    '@type': 'WebSite', '@id': `${SITE_URL}/#website`, url: `${SITE_URL}/`,
    name: 'DAMASCUS', creator: { '@id': personId }, about: { '@id': softwareId },
  }
  return { '@context': 'https://schema.org', '@graph': [person, software, entity] }
}
