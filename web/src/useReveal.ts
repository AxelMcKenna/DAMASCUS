import { useEffect, useRef } from 'react'

// Adds the `in` class when the element scrolls into view, triggering the
// `reveal` CSS animation once.
export function useReveal(options: IntersectionObserverInit = {}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || !('IntersectionObserver' in window) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    el.classList.add('reveal-pending')
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('in')
            io.unobserve(e.target)
          }
        }
      },
      { threshold: 0.15, ...options },
    )
    io.observe(el)
    return () => {
      io.disconnect()
      el.classList.remove('reveal-pending')
    }
  }, [])
  return ref
}
