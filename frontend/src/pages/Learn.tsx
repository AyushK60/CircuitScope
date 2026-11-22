import { useState } from "react";
import GlossaryList from "../components/GlossaryList";
import Feedforward101 from "./learn/Feedforward101";
import TopicDiagram from "./learn/Diagrams";
import { CHAPTERS, DEPTHS, TOPICS, type Depth } from "./learn/topics";
import { useActiveSection } from "./report/useActiveSection";

/**
 * Learn: a short course in four chapters. The depth switch (simple, more context, technical) stays in view
 * beside the text. These pages describe how the app works; every measured value elsewhere comes from the
 * running model, not from here.
 */

const FF_CHAPTER = "A contrast";
const scrollToId = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

export default function Learn() {
  const [depth, setDepth] = useState<Depth>("context");
  const ids = [...TOPICS.map((t) => `ln-${t.id}`), "ln-ff101"];
  const active = useActiveSection(ids);

  return (
    <main className="page ln-page">
      <div className="ln-layout">
        <aside className="ln-side" aria-label="Contents">
          <small>Explain it</small>
          <div className="ln-depth" role="tablist" aria-label="Explanation depth">
            {DEPTHS.map((d) => (
              <button
                key={d.id}
                role="tab"
                aria-selected={depth === d.id}
                className={depth === d.id ? "on" : ""}
                onClick={() => setDepth(d.id)}
              >
                <b>{d.label}</b>
                <span>{d.hint}</span>
              </button>
            ))}
          </div>
          <small>Chapters</small>
          <nav>
            {CHAPTERS.map((c, ci) => (
              <div className="ln-ch" key={c}>
                <h4>
                  {ci + 1} · {c}
                </h4>
                {TOPICS.filter((t) => t.chapter === ci).map((t) => (
                  <button key={t.id} className={active === `ln-${t.id}` ? "on" : ""} onClick={() => scrollToId(`ln-${t.id}`)}>
                    {t.term}
                  </button>
                ))}
              </div>
            ))}
            <div className="ln-ch">
              <h4>
                5 · {FF_CHAPTER}
              </h4>
              <button className={active === "ln-ff101" ? "on" : ""} onClick={() => scrollToId("ln-ff101")}>
                A plain network
              </button>
            </div>
          </nav>
        </aside>

        <div className="ln-main">
          <p className="ex-kick">A short course in how GPT-2 works</p>
          <h1 className="ex-title">Learn</h1>
          <p className="ex-sub">
            Every idea the app uses, in four chapters, at the depth you pick. This page explains how the app works;
            every measured number elsewhere comes from the running model, not from this page.
          </p>

          <div className="ln-isnt">
            <div>
              <b>CircuitScope is</b>A diagram of a real transformer, live token-level measurements from named
              hooks, and controlled activation-patching experiments with reproducible numbers.
            </div>
            <div>
              <b>CircuitScope is not</b>A claim that the model&apos;s complete reasoning is visible, a proof that any single
              head &quot;understands&quot; a concept, or a statistical study. An attention heatmap is a measurement; a patch
              result is evidence under that specific intervention.
            </div>
          </div>

          {CHAPTERS.map((c, ci) => (
            <section key={c} aria-label={`Chapter ${ci + 1}: ${c}`}>
              <p className="ln-chap">
                Chapter {ci + 1} · {c}
              </p>
              {TOPICS.filter((t) => t.chapter === ci).map((t) => (
                <article className="ln-topic" id={`ln-${t.id}`} key={t.id}>
                  <div>
                    <h2>{t.term}</h2>
                    <p>{t[depth]}</p>
                    <a className="ln-live" href={t.live.href}>
                      See it live: {t.live.label} →
                    </a>
                  </div>
                  <div className="ln-fig">
                    <TopicDiagram id={t.id} />
                  </div>
                </article>
              ))}
            </section>
          ))}

          <section aria-label={`Chapter 5: ${FF_CHAPTER}`}>
            <p className="ln-chap">Chapter 5 · {FF_CHAPTER}</p>
            <article className="ln-topic" id="ln-ff101">
              <div>
                <h2>Neural Networks 101: a plain network</h2>
                <p>
                  <strong>Not GPT-2.</strong> This classic dense stack passes values forward layer by layer with no memory of
                  position and no shortcut paths. GPT-2 instead tokenizes text, uses attention so every token can look at
                  earlier tokens, and writes every result into one shared <em>residual stream</em>. Compare it with the Model
                  Explorer&apos;s map.
                </p>
                <a className="ln-live" href="#/lab">
                  Try one yourself: the Neural Network Lab →
                </a>
              </div>
              <div className="ln-fig">
                <Feedforward101 />
              </div>
            </article>
          </section>

          <details className="ln-glossary">
            <summary>All terms (glossary)</summary>
            <GlossaryList level={depth === "simple" ? "curious" : depth === "context" ? "student" : "expert"} />
          </details>
        </div>
      </div>
    </main>
  );
}
