import assert from 'node:assert/strict'
import { readFile, access } from 'node:fs/promises'
import test from 'node:test'
import { preview } from 'vite'

const origin = 'https://damascus-llm.dev'
const routes = ['/', '/axel-mckenna', '/writing/building-damascus']
const load = path => readFile(path === '/' ? 'dist/index.html' : `dist${path}.html`, 'utf8')
const schemas = html => [...html.matchAll(/<script[^>]+type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/g)].map(match => JSON.parse(match[1]))

for (const path of routes) {
  test(`${path} has crawlable content and unique production metadata`, async () => {
    const html = await load(path)
    assert.equal((html.match(/<h1\b/g) ?? []).length, 1)
    assert.match(html, /Axel McKenna/)
    assert.equal((html.match(/<title>/g) ?? []).length, 1)
    assert.equal((html.match(/rel="canonical"/g) ?? []).length, 1)
    assert.ok(html.includes(`rel="canonical" href="${origin}${path}"`))
    assert.ok(html.includes(`property="og:url" content="${origin}${path}"`))
    assert.match(html, /name="description" content="[^"]{60,}"/)
    assert.match(html, /name="twitter:card" content="summary_large_image"/)
    assert.doesNotMatch(html, /noindex|localhost|127\.0\.0\.1/)
    assert.ok(html.includes('href="/axel-mckenna"'))
    assert.ok(html.includes('href="/writing/building-damascus"'))
    assert.equal(schemas(html).length, 1)
    for (const match of html.matchAll(/(?:src|href)="(\/assets\/[^"?#]+)"/g)) {
      await access(`dist${match[1]}`)
    }
  })
}

test('profile and article reference the same author with visible attribution', async () => {
  const creator = await load('/axel-mckenna')
  const article = await load('/writing/building-damascus')
  const personId = `${origin}/axel-mckenna#person`
  const profile = schemas(creator)[0]['@graph'].find(node => node['@type'] === 'ProfilePage')
  const technical = schemas(article)[0]['@graph'].find(node => node['@type'] === 'TechArticle')
  assert.equal(profile.mainEntity['@id'], personId)
  assert.equal(technical.author['@id'], personId)
  assert.equal(technical.author.url, `${origin}/axel-mckenna`)
  assert.match(article, /rel="author">Axel McKenna<\/a>/)
  assert.match(article, /1\.33e-2/)
  assert.match(article, /not throughput benchmarks/)
  assert.match(article, /has not been established/)
  assert.match(creator, /https:\/\/www\.liquorfy\.co\.nz\/about/)
})

test('robots and sitemap expose exactly the public pages; 404 stays out', async () => {
  const robots = await readFile('dist/robots.txt', 'utf8')
  const sitemap = await readFile('dist/sitemap.xml', 'utf8')
  assert.match(robots, /Allow: \//)
  assert.ok(robots.includes(`Sitemap: ${origin}/sitemap.xml`))
  assert.deepEqual([...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map(match => match[1]), routes.map(path => `${origin}${path}`))
  const missing = await readFile('dist/404.html', 'utf8')
  assert.match(missing, /name="robots" content="noindex, follow"/)
  assert.doesNotMatch(missing, /rel="canonical"/)
  assert.match(missing, /<h1[^>]*>Page not found<\/h1>/)
})

test('static headings do not depend on JavaScript to become visible', async () => {
  const cssFiles = [...(await load('/')).matchAll(/href="(\/assets\/[^"?#]+\.css)"/g)]
  const css = (await Promise.all(cssFiles.map(match => readFile(`dist${match[1]}`, 'utf8')))).join('\n')
  assert.doesNotMatch(css, /\.reveal\{opacity:0/)
  assert.match(css, /\.reveal\.reveal-pending/)
})

test('extensionless public URLs serve the correct prerendered page over HTTP', async () => {
  const server = await preview({ preview: { host: '127.0.0.1', port: 0, open: false } })
  try {
    const address = server.httpServer.address()
    for (const path of routes) {
      const response = await fetch(`http://127.0.0.1:${address.port}${path}`)
      assert.equal(response.status, 200)
      assert.equal(await response.text(), await load(path))
    }
    for (const path of ['/robots.txt', '/sitemap.xml', '/llms.txt']) {
      const response = await fetch(`http://127.0.0.1:${address.port}${path}`)
      assert.equal(response.status, 200)
      assert.equal(await response.text(), await readFile(`dist${path}`, 'utf8'))
    }
  } finally {
    await new Promise((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()))
  }
})

test('AI discovery summary is linked and distinguishes verified results from goals', async () => {
  const summary = await readFile('dist/llms.txt', 'utf8')
  assert.match(summary, /^# DAMASCUS/)
  assert.match(summary, /created by Axel McKenna/)
  assert.match(summary, /1\.33e-2/)
  assert.match(summary, /target, not an achieved benchmark/)
  const robots = await readFile('dist/robots.txt', 'utf8')
  for (const agent of ['OAI-SearchBot', 'Claude-SearchBot']) assert.ok(robots.includes(`User-agent: ${agent}\nAllow: /`))
  for (const path of routes) assert.match(await load(path), /rel="describedby" href="\/llms.txt"/)
})
