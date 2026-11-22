// The Ranking page, Phase 2 panels (M4): browse the corpus by K-means topic cluster, how good the clustering is against an automatic
// labelling, and what inexact top-K retrieval saves and costs. Everything shown comes from the frozen files written by
// `python -m m4_rank.cluster build` and `python -m m4_rank.inexact report`; when a file is missing the panel says how to make it.
import { api } from "./api.js";
import { $, f2, f3, formatDate, h, plural } from "./dom.js";
import { animateIn } from "./motion.js";
import { openReader } from "./reader.js";

const pct0 = (x) => (typeof x === "number" ? `${Math.round(x * 100)}%` : "n/a");

function missing(title, message) {
  return h("section", { class: "card" }, [h("h3", {}, title), h("div", { class: "empty" }, [h("strong", {}, "Not built yet"), message])]);
}

async function showMembers(box, c) {
  box.replaceChildren(h("p", { class: "note" }, "Loading"));
  try {
    const data = await api.cluster(c.cluster_id, 15);
    const list = data.members.map((m) => ({ doc_id: m.doc_id, title: m.title }));
    box.replaceChildren(
      h("h4", {}, [`Cluster ${data.cluster_id}: ${data.label}`, h("small", { class: "note" }, `  ${plural(data.size, "judgment")}, closest to the centroid first`)]),
      h("ol", { class: "member-list" }, data.members.map((m) => h("li", {}, [
        h("button", { class: "link-btn", type: "button", onclick: () => openReader(m.doc_id, m.title || "", "", { list }) }, m.title || m.doc_id),
        h("span", { class: "note" }, `  ${m.date ? formatDate(m.date) : ""}  ·  distance ${f3(m.distance)}`),
      ]))),
    );
    animateIn(box.children, { stagger: 0.02 });
  } catch (err) {
    box.replaceChildren(h("p", { class: "callout callout-error" }, err.message));
  }
}

function clusterPanel(data) {
  if (!data.available) return missing("Browse by topic cluster", data.message);
  const ev = data.evaluation;
  const members = h("div", { class: "cluster-members" }, h("p", { class: "note" }, "Choose a cluster to see its most typical judgments."));
  const grid = h("div", { class: "cluster-grid" }, data.clusters.map((c) => {
    const btn = h("button", { class: "cluster-card", type: "button" }, [
      h("span", { class: "cluster-id" }, `#${c.cluster_id}`),
      h("strong", {}, c.label),
      h("span", { class: "note" }, `${plural(c.size, "judgment")}${c.top_offence ? `  ·  ${c.top_offence.replace(/^OFF_/, "").replaceAll("_", " ").toLowerCase()} ${pct0(c.top_offence_share)}` : ""}`),
    ]);
    btn.addEventListener("click", () => {
      grid.querySelectorAll(".cluster-card").forEach((b) => b.classList.toggle("is-on", b === btn));
      showMembers(members, c);
    });
    return btn;
  }));
  const cross = data.crosscheck;
  return h("section", { class: "card" }, [
    h("h3", {}, ["Browse by topic cluster ", h("span", { class: "mod mod-m4" }, "K-means")]),
    h("p", { class: "note" }, `K = ${data.k} (${data.k_rule === "elbow" ? "elbow of the RSS curve" : "fixed"}; AIC would pick ${data.k_aic}) over ${data.n_docs} judgments as tf-idf vectors of ${data.dims} terms; k-means++ seeding, ${data.n_init} restarts, seed ${data.seed}. Labels are the centroid's most distinctive terms.`),
    grid,
    members,
    h("h4", {}, "How well the clusters line up with the dominant offence"),
    h("p", { class: "note" }, `Against an automatic labelling (each judgment's most-referenced offence from M2's statute references), not a human gold standard. ${ev.labelled_docs} judgments labelled, ${ev.unlabelled_docs} without an offence left out.`),
    h("div", { class: "table-wrap" }, h("table", { class: "table" }, [
      h("thead", {}, h("tr", {}, ["purity", "NMI", "Rand index", "F1", "F5", "RSS"].map((x) => h("th", {}, x)))),
      h("tbody", {}, h("tr", {}, [f3(ev.purity), f3(ev.nmi), f3(ev.rand_index), f3(ev.f1), f3(ev.f5), f2(data.rss)].map((x) => h("td", {}, x)))),
    ])),
    cross ? h("p", { class: "note" }, `Cross-check: scikit-learn's KMeans reaches RSS ${f2(cross.sklearn_rss)} (ours ${f2(cross.our_rss)}); its NMI of our clusters is ${f3(cross.sklearn_nmi_of_our_clusters)} (ours ${f3(cross.our_nmi)}).`) : null,
  ]);
}

function inexactPanel(data) {
  if (!data.available) return missing("Inexact top-K: speed vs quality", data.message);
  const rows = data.summary;
  const s = data.settings;
  return h("section", { class: "card" }, [
    h("h3", {}, ["Inexact top-K: speed vs quality ", h("span", { class: "mod mod-m4" }, "dev queries")]),
    h("p", { class: "note" }, `Each method picks a smaller set of candidates and scores only those with the same zone-weighted BM25 as M1's search(); top-${data.k}, ${data.queries.length} judged dev queries, BM25 ranking only. Champion r = ${s.champion_r}; tier 1 = authority in the top ${Math.round((1 - s.tier_cut) * 100)}% or a bench of ${s.tier1_min_bench}+.`),
    h("div", { class: "table-wrap" }, h("table", { class: "table" }, [
      h("thead", {}, h("tr", {}, ["method", "median ms", "speed-up", "judgments scored", "work vs exact", "overlap@10", "nDCG@10", "change", "judged@10"].map((x) => h("th", {}, x)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, [
        r.method, f2(r.median_ms), `${f2(r.speedup)}x`, String(Math.round(r.mean_candidates)), pct0(r.work_ratio),
        f2(r["mean_overlap@10"]), f3(r["mean_nDCG@10"]), r.nDCG_change == null ? "n/a" : (r.nDCG_change >= 0 ? "+" : "") + f3(r.nDCG_change), f2(r["mean_judged@10"]),
      ].map((x) => h("td", {}, x))))),
    ])),
    h("p", { class: "note" }, "The qrels were pooled from exact systems, so a method that surfaces unjudged judgments is scored as if they were irrelevant: read the nDCG change next to judged@10. None of these is the default ranking; the tiered index is the full_tiered ablation."),
  ]);
}

export async function renderPhase2() {
  const root = $("#phase2-body");
  if (!root) return;
  root.replaceChildren(h("p", { class: "note" }, "Loading the clustering and inexact top-K panels"));
  const [clusters, inexact] = await Promise.allSettled([api.clusters(), api.inexact()]);
  const blocks = [
    clusters.status === "fulfilled" ? clusterPanel(clusters.value) : missing("Browse by topic cluster", clusters.reason.message),
    inexact.status === "fulfilled" ? inexactPanel(inexact.value) : missing("Inexact top-K: speed vs quality", inexact.reason.message),
  ];
  root.replaceChildren(h("h2", { class: "section-title" }, "Clustering and inexact retrieval"), ...blocks);
  animateIn(root.children, { stagger: 0.05 });
}
