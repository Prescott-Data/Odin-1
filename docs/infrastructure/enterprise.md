---
icon: material/office-building
---

# Odin Enterprise

**Odin-1** is open source under the MIT license. You can self-host it, run it on any infrastructure, and take it to production yourself, forever, for free. Many teams do exactly that.

**Odin-2** is the proprietary next generation of Odin, operated and supported by Prescott Data. It is built for teams that have proven the idea with Odin-1 and now need to run graph intelligence across many graphs, under strict security and governance, with domain-tuned accuracy and a supported platform underneath. You are never forced to upgrade; Odin-2 exists for when single-graph, self-operated retrieval is no longer enough.

---

## The Odin family

Odin is built as a family, not a single product. Every generation shares one engine and one way of thinking; what changes is how far it reaches and who operates it.

- **Odin-1 is the open base.** MIT-licensed, free, and built to spread. The community adopts it, builds on it, and takes it in directions no roadmap could plan for. Its path belongs to its users as much as to us, and it stays open.
- **Odin-2 and the generations after it are the managed tier.** Each new generation adds capabilities that only make sense with an operator behind them, federation, zero-trust security, and domain intelligence, and ships them as a supported platform rather than a library.

The open base and the managed generations reinforce each other rather than compete. Starting on Odin-1 costs you nothing and locks you into nothing, and when you need to run graph intelligence in production at organizational scale, the path onward is already built.

---

## Odin-1 vs Odin-2

|  | Odin-1 (OSS) | Odin-2 (Enterprise) |
|---|---|---|
| **License** | MIT, free forever | Commercial agreement |
| **Graph scope** | Single knowledge graph | Federated, many graphs at once |
| **Backends** | ArangoDB, JanusGraph, custom adapters | Managed multi-graph accessor |
| **Security** | You configure it | Zero-trust Nexus sessions, tenant isolation, clearance levels |
| **Access governance** | None | Odin Registry, credential-free scoped access |
| **Domain accuracy** | General-purpose | Domain Intelligence Service (DIS) |
| **Link discovery** | Paths over existing edges | Phantom-edge prediction of missing links |
| **Priors** | Learned once from your graph | Adaptive priors that learn online from agent feedback |
| **Operation** | Self-managed | Fully managed by Prescott Data |
| **Support** | Community (GitHub Issues) | Dedicated engineering with an SLA |
| **Graph & ontology** | You build it | Engineered for you (professional services) |

---

## What Odin-2 adds

### Federated multi-graph intelligence

Odin-1 explores one graph at a time. Odin-2 moves from single-graph to **federated** retrieval: a multi-graph accessor lets agents reason across many knowledge graphs as if they were one, with boundary delegation that follows connections from a graph you own into graphs you are permitted to see. This is the difference between answering questions inside a dataset and answering questions across an organization.

### Domain Intelligence Service (DIS)

General-purpose scoring treats every domain the same. The **Domain Intelligence Service** layers domain-specific intelligence over retrieval, so the engine understands what "important", "plausible", and "surprising" mean in *your* field (healthcare, finance, supply chain, compliance) rather than in the abstract. It is the difference between a competent generalist and a specialist.

### Zero-trust security with Nexus

Odin-2 secures every retrieval through **Nexus**, a zero-trust session layer. Agents never hold raw credentials; they operate inside ephemeral, scoped sessions issued per task. **On-Behalf-Of (OBO) sessions** carry a user's identity, tenant, and data-clearance level through the whole retrieval, so an agent only ever sees the graph a given user is cleared to see. Multi-tenant isolation and data-clearance levels are enforced at the session boundary, not bolted on afterward.

### Governed access with the Odin Registry

The **Odin Registry**, backed by Nexus, is the control plane for what agents can reach: which graphs, which tools, which scopes. Access is granted centrally and audited, so security and platform teams get governance and a clear audit trail without slowing developers down.

### Advanced retrieval capabilities

Odin-2 extends the core engine with capabilities aimed at production discovery work:

- **Phantom-edge prediction** surfaces links that are missing from the data but plausible given its structure, so agents can investigate connections that were never explicitly recorded.
- **Motif-aware adaptive priors** learn from agent feedback over time, sharpening what the engine treats as high-signal as your team uses it.
- **Structural memory** lets the engine recognize and reuse patterns it has seen before, keeping large-scale exploration fast and consistent.

---

## Managed deployment and support

With Odin-2, Prescott Data provisions, operates, and supports the full stack on your chosen cloud or inside your private network:

- Zero-ops onboarding: your team connects to a running system, not a setup guide.
- Patching, upgrades, scaling, backup, and disaster recovery handled for you.
- Private networking, VPC peering, and on-premises options; your data stays in your environment.
- Dedicated engineering support with a response-time SLA, plus onboarding, architecture review, and migration help.

---

## Professional services

Most of the value in an enterprise deployment is won or lost before retrieval ever runs, in the work of turning raw data into a knowledge graph worth reasoning over. Prescott Data does that work with you.

### Knowledge graph engineering

We design and build your graph from the ground up: the ontology, the entity and relationship schemas, and the extraction pipelines that populate them from your documents, databases, and systems of record. You bring the domain and the data; we deliver a graph that is ready for Odin to reason over.

### Ontology design

We work with your domain experts to encode how your field actually thinks: the entities, the relationships, and the rules that make a connection meaningful, so the graph reflects your business rather than a generic template. A good ontology is the difference between a graph that answers real questions and one that merely stores data.

### Domain tuning

We calibrate the Domain Intelligence Service to your field so that importance, plausibility, and surprise mean what they should mean in your data, not in the abstract.

### Onboarding and migration

We move your Odin-1 proof of concept onto Odin-2, connect your data sources, and validate the result against the outcomes you proved in the POC.

### Architecture review

We review your agent and retrieval architecture with you and help you design for scale, security, and cost before they become problems.

The deliverable is not just software. It is a working graph-intelligence capability, built for your domain and operated on your behalf.

---

## The upgrade path

The path is intentionally smooth. Prove the value on **Odin-1** with a proof of concept on your own graph, then move that same mental model (seeds, retrieval, triage, agent integration) onto **Odin-2** when you need federation, security, domain tuning, and a managed platform. Nothing you learn with Odin-1 is wasted; Odin-2 is the same engine, scaled up and hardened for production.

---

## Talk to us

If you have validated with Odin-1 and want to take it into production at organizational scale, reach out.

<div class="jc-cta" markdown>
[Contact Enterprise](mailto:enterprise@prescottdata.io){ .jc-btn }
[Prescott Data](https://prescottdata.io){ .jc-btn .jc-btn-github target="_blank" rel="noopener" }
</div>
