"""
FusionKitchen — Theme
=======================
Grindhouse/pulp-cinema styling: warm black background, blood red + marquee
yellow accents, poster typography. Pure presentation — no logic lives here.
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;500;600;700&family=Courier+Prime:wght@400;700&display=swap');

:root {
  --void: #0D0B0A;
  --smoke: #1C1815;
  --blood: #A6192E;
  --marquee: #E8B923;
  --bone: #F2EDE4;
  --rust: #6E2A1E;
}

.gradio-container {
  background: radial-gradient(ellipse at top, #1a1512 0%, var(--void) 65%) !important;
  background-image:
    radial-gradient(ellipse at top, #1a1512 0%, var(--void) 65%),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, transparent 1px, transparent 2px);
  font-family: 'Oswald', sans-serif !important;
  color: var(--bone) !important;
}

.gradio-container .block,
.gradio-container .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
footer, .gradio-container footer { opacity: 0.6; }

@media (prefers-reduced-motion: no-preference) {
  .hero-title { animation: flicker-in 1.1s ease-out; }
  .recipe-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
  .recipe-card:hover { transform: rotate(-0.6deg) translateY(-4px); box-shadow: 8px 12px 0 rgba(0,0,0,0.5); }
}
@keyframes flicker-in {
  0% { opacity: 0; }
  15% { opacity: 0.9; } 20% { opacity: 0.2; } 30% { opacity: 1; }
  45% { opacity: 0.6; } 55% { opacity: 1; }
  100% { opacity: 1; }
}

.hero-wrap { text-align: center; padding: 2.2rem 1rem 1.2rem; }
.hero-title {
  font-family: 'Anton', sans-serif;
  font-size: clamp(1.55rem, 8.5vw, 4.6rem);
  color: var(--blood);
  text-shadow: 3px 3px 0 var(--marquee), 6px 6px 0 rgba(0,0,0,0.6);
  letter-spacing: 0.02em;
  transform: rotate(-1.5deg);
  margin: 0;
  line-height: 1;
}
.hero-sub {
  font-family: 'Oswald', sans-serif;
  font-weight: 600;
  letter-spacing: 0.28em;
  color: var(--bone);
  opacity: 0.85;
  font-size: 0.85rem;
  margin-top: 0.9rem;
  text-transform: uppercase;
}
.hero-rule { width: 140px; height: 3px; background: var(--marquee); margin: 1rem auto 0; }

.chapter-header { padding: 0.5rem 0.2rem 1rem; }
.eyebrow {
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  color: var(--marquee);
  letter-spacing: 0.3em;
  font-size: 0.78rem;
}
.chapter-header h2 {
  font-family: 'Anton', sans-serif;
  color: var(--bone);
  font-size: clamp(1.6rem, 4vw, 2.3rem);
  margin: 0.2rem 0 0;
  letter-spacing: 0.01em;
}
.rule { width: 100%; height: 2px; background: linear-gradient(90deg, var(--rust), transparent); margin: 0.6rem 0; }
.chapter-sub { font-size: 0.92rem; color: var(--bone); opacity: 0.82; margin: 0; }
.chapter-sub em { color: var(--marquee); font-style: normal; }

.intake-label {
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  color: var(--marquee);
  letter-spacing: 0.15em;
  font-size: 0.85rem;
  display: block;
  margin-bottom: 0.4rem;
}
#ingredient-input textarea, #ingredient-input input {
  background: var(--smoke) !important;
  border: 2px solid var(--rust) !important;
  color: var(--bone) !important;
  font-family: 'Courier Prime', monospace !important;
  font-size: 1rem !important;
  border-radius: 2px !important;
}
#ingredient-input textarea:focus, #ingredient-input input:focus {
  border-color: var(--marquee) !important;
  box-shadow: 0 0 0 2px rgba(232,185,35,0.25) !important;
}

#find-btn {
  font-family: 'Anton', sans-serif !important;
  font-size: 1.1rem !important;
  letter-spacing: 0.04em !important;
  background: var(--blood) !important;
  color: var(--bone) !important;
  border: 2px solid var(--rust) !important;
  border-radius: 2px !important;
  box-shadow: 4px 4px 0 rgba(0,0,0,0.5) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#find-btn:hover {
  transform: translate(-2px, -2px) !important;
  box-shadow: 6px 6px 0 rgba(0,0,0,0.5) !important;
  background: #8f1527 !important;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.4rem;
  margin-top: 0.5rem;
}
.recipe-card {
  position: relative;
  background: var(--smoke);
  border: 1px solid var(--rust);
  border-left: 4px solid var(--blood);
  padding: 1.3rem 1.3rem 1.1rem;
  border-radius: 2px;
  box-shadow: 5px 5px 0 rgba(0,0,0,0.35);
}
.card-title {
  font-family: 'Anton', sans-serif;
  color: var(--marquee);
  font-size: 1.35rem;
  margin: 0 2.6rem 0.6rem 0;
  line-height: 1.15;
  text-transform: uppercase;
}
.match-badge {
  position: absolute;
  top: -14px; right: -10px;
  background: var(--blood);
  color: var(--bone);
  border: 2px solid var(--void);
  border-radius: 50%;
  width: 62px; height: 62px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-family: 'Anton', sans-serif;
  font-size: 1.05rem;
  transform: rotate(8deg);
  box-shadow: 2px 3px 0 rgba(0,0,0,0.5);
}
.match-badge span { font-family: 'Courier Prime', monospace; font-size: 0.5rem; letter-spacing: 0.05em; }

.stamp {
  display: inline-block;
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  line-height: 1.3;
  padding: 0.35rem 0.55rem;
  border: 2px dashed;
  border-radius: 3px;
  transform: rotate(-3deg);
  margin: 0.3rem 0 0.8rem;
  text-align: center;
}
.stamp-unverified { color: var(--blood); border-color: var(--blood); }
.stamp-verified { color: #6FA37A; border-color: #6FA37A; }

.caution-tag {
  display: inline-block;
  background: var(--blood);
  color: var(--bone);
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.64rem;
  letter-spacing: 0.05em;
  padding: 0.25rem 0.55rem;
  border-radius: 2px;
  margin-left: 0.5rem;
}

.match-label {
  display: block;
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: var(--marquee);
  opacity: 0.85;
  margin: -0.3rem 0 0.5rem;
}

.overlap-line {
  font-size: 0.82rem;
  line-height: 1.4;
  color: var(--bone);
  opacity: 0.85;
  margin-bottom: 0.3rem;
}
.overlap-label {
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
  color: var(--marquee);
  margin-right: 0.3rem;
}

.ingredients-line {
  font-family: 'Courier Prime', monospace;
  font-size: 0.78rem;
  color: var(--bone);
  opacity: 0.7;
  border-top: 1px solid var(--rust);
  border-bottom: 1px solid var(--rust);
  padding: 0.5rem 0;
  margin-bottom: 0.7rem;
}

.field { font-size: 0.88rem; margin-bottom: 0.55rem; line-height: 1.4; color: var(--bone); opacity: 0.92; }
.field-label {
  display: block;
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  color: var(--marquee);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  margin-bottom: 0.15rem;
}

.footer-note {
  text-align: center;
  font-family: 'Courier Prime', monospace;
  font-size: 0.72rem;
  opacity: 0.55;
  margin: 2.2rem 0 1rem;
  letter-spacing: 0.03em;
}
"""
