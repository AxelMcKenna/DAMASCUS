import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createServer } from 'vite'

// Render the same React tree used by the browser, without a running server or API.
const server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom' })
try {
  const { render, pages, SITE_URL } = await server.ssrLoadModule('/src/entry-server.tsx')
  const template = await readFile('dist/index.html', 'utf8')
  if (!template.includes('<!--seo:start-->') || !template.includes('<div id="root"></div>')) {
    throw new Error('Prerender template markers are missing')
  }
  for (const path of [...pages.map(page => page.path), '/404']) {
    const { head, html } = render(path)
    const filename = path === '/' ? 'dist/index.html' : `dist${path}.html`
    await mkdir(dirname(filename), { recursive: true })
    await writeFile(filename, template
      .replace(/<!--seo:start-->[\s\S]*?<!--seo:end-->/, () => `<!--seo:start-->\n    ${head}\n    <!--seo:end-->`)
      .replace('<div id="root"></div>', () => `<div id="root">${html}</div>`))
  }
  await writeFile('dist/sitemap.xml', `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${pages.map(page => `  <url><loc>${SITE_URL}${page.path}</loc></url>`).join('\n')}\n</urlset>\n`)
  console.log(`Prerendered ${pages.length} public pages and a 404 page.`)
} finally {
  await server.close()
}
