import { renderToString } from 'react-dom/server'
import App from './App'
import { getPage, pages, structuredData, SITE_URL } from './seo'

export { pages, SITE_URL }

const escape = (value: string) => value.replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[char]!)

export function render(path: string) {
  const page = getPage(path)
  const url = `${SITE_URL}${page.path}`
  const image = `${SITE_URL}/img/damascus.png`
  const head = [
    `<title>${escape(page.title)}</title>`,
    `<meta name="description" content="${escape(page.description)}" />`,
    '<meta name="author" content="Axel McKenna" />',
    `<meta name="robots" content="${page.path === '/404' ? 'noindex, follow' : 'index, follow'}" />`,
    ...(page.path === '/404' ? [] : [`<link rel="canonical" href="${url}" />`]),
    `<meta property="og:type" content="${page.type}" />`,
    '<meta property="og:site_name" content="DAMASCUS" />',
    `<meta property="og:title" content="${escape(page.title)}" />`,
    `<meta property="og:description" content="${escape(page.description)}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:image" content="${image}" />`,
    '<meta property="og:image:alt" content="Damascus steel texture" />',
    '<meta name="twitter:card" content="summary_large_image" />',
    `<meta name="twitter:title" content="${escape(page.title)}" />`,
    `<meta name="twitter:description" content="${escape(page.description)}" />`,
    `<meta name="twitter:image" content="${image}" />`,
    ...(page.path === '/404' ? [] : [`<script type="application/ld+json">${JSON.stringify(structuredData(path)).replace(/</g, '\\u003c')}</script>`]),
  ].join('\n    ')
  return { head, html: renderToString(<App path={path} />) }
}
