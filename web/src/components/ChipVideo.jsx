import { useEffect, useRef } from 'react'

// Scrubs the chip-expansion video to scroll position: as the section travels
// up through the viewport, playback advances from assembled (frame 0) to fully
// exploded (final frame). Falls back to the end frame under reduced-motion.
export default function ChipVideo() {
  const wrap = useRef(null)
  const video = useRef(null)
  const target = useRef(0)

  useEffect(() => {
    const v = video.current
    const el = wrap.current
    if (!v || !el) return

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let raf = 0
    let current = 0
    let duration = 0

    const onMeta = () => {
      duration = v.duration || 0
    }
    v.addEventListener('loadedmetadata', onMeta)

    const computeProgress = () => {
      const r = el.getBoundingClientRect()
      const vh = window.innerHeight
      const start = vh * 0.92 // enters from the bottom
      const end = vh * 0.32 // fully exploded near center/top
      return Math.max(0, Math.min(1, (start - r.top) / (start - end)))
    }

    // Ease the seek toward the scroll target so scrubbing feels smooth.
    const tick = () => {
      if (duration) {
        current += (target.current - current) * 0.15
        const t = current * duration
        if (Math.abs(v.currentTime - t) > 0.01) v.currentTime = t
      }
      raf = requestAnimationFrame(tick)
    }

    const onScroll = () => {
      target.current = computeProgress()
    }

    if (reduce) {
      v.addEventListener('loadeddata', () => {
        if (v.duration) v.currentTime = v.duration
      })
      return () => v.removeEventListener('loadedmetadata', onMeta)
    }

    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    raf = requestAnimationFrame(tick)

    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      cancelAnimationFrame(raf)
      v.removeEventListener('loadedmetadata', onMeta)
    }
  }, [])

  return (
    <div ref={wrap} className="relative">
      <div className="absolute -inset-10 bg-forge-600/12 blur-3xl" />
      <div className="relative border border-steel-800 bg-steel-900/40">
        <div className="flex items-center justify-between border-b border-steel-800 px-4 py-2.5 font-mono text-xs text-steel-500">
          <span>silicon.expand</span>
          <span>scroll ▾</span>
        </div>
        <video
          ref={video}
          className="w-full"
          src="/img/chip-expand.mp4"
          poster="/img/chip-normal.png"
          muted
          playsInline
          preload="auto"
          tabIndex={-1}
        />
      </div>
    </div>
  )
}
