# benefits-plan-qa

Ask plain-English questions of any benefits plan PDF — get answers grounded in the document, with page citations. Built with Claude (Sonnet 4) via OpenRouter.

## What it does

Drop in a Summary of Benefits and Coverage (SBC), SPD, or similar plan document, ask a question like *"What's the out-of-pocket max for a family?"*, and get back a one-paragraph answer plus a `Sources: page 3` line showing exactly where in the document the answer came from. The tool reads the PDF, splits it into page-aware chunks, embeds them locally with `sentence-transformers`, retrieves the most relevant passages with FAISS, and asks Claude to answer using only those passages.

## Why I built it

I've spent 10 years in benefits consulting (WTW, Keenan/Gallagher). Consultants and HR admins constantly need to find specific coverage details inside long plan documents — and Ctrl-F doesn't work because the language varies by carrier. One plan says "myocardial infarction," another says "heart attack." One says "Tier 1 generic," another says "preferred generic." Members ask in plain English; the doc speaks in legalese.

This is the kind of small, ugly, daily problem where a Claude-based assistant earns its keep. Three minutes saved per question, fifteen questions a week, across a few hundred consultants — that's real time back. The repo is a working proof of the pattern, not a finished product.

## Quickstart

```bash
git clone https://github.com/becastil/benefits-plan-qa.git
cd benefits-plan-qa

# create a venv (Python 3.10+ recommended)
python3 -m venv .venv && source .venv/bin/activate

# install deps
pip install -e .

# set your OpenRouter key (get one at https://openrouter.ai/keys)
cp .env.example .env
# then edit .env and paste your key

# ask a question against the bundled CMS sample SBC
python main.py samples/sample-sbc.pdf "What is the overall deductible for an individual?"
```

You can swap in any PDF benefits document for `samples/sample-sbc.pdf`.

## Example Q&A

All output below is **actual, unedited** CLI output produced by running `main.py` against the bundled `samples/sample-sbc.pdf` (the CMS federal sample SBC). Reproducible from a clean clone.

```
Q: What is the overall deductible for an individual?
The overall deductible for an individual is $500.

Sources: page 1
```

```
Q: What is the out-of-pocket maximum for a family?
The out-of-pocket maximum for a family is $5,000 for network providers
and $8,000 for out-of-network providers.

Sources: page 1
```

```
Q: Are preventive care services covered without paying the deductible?
Yes, preventive care services are covered before you meet your deductible
with no charge for in-network providers. However, you should ask your
provider if the services are actually preventive, as you may have to pay
for services that aren't considered preventive.

Sources: page 1, page 2
```

```
Q: What does it cost to see a primary care doctor for a non-preventive visit?
For a non-preventive primary care visit, you'll pay a $35 copay per
office visit if you see a network provider, or 40% coinsurance if you
see an out-of-network provider. The deductible does not apply to these visits.

Sources: page 2
```

```
Q: How much do generic drugs cost under this plan?
Generic drugs (Tier 1) have a $10 copay per prescription for both retail
and mail order when using in-network providers, and 40% coinsurance when
using out-of-network providers.

Sources: page 2
```

Notice how the answer to question 4 picked up the *condition* that the deductible doesn't apply to PCP visits — something you'd miss with Ctrl-F because it's expressed as a row in a table, not a sentence.

## How it works

1. `pypdf` extracts the text from each page of the PDF, preserving page numbers.
2. Each page is split into overlapping ~1,200-character chunks; the page number stays attached to every chunk.
3. Chunks are embedded locally with the `sentence-transformers/all-MiniLM-L6-v2` model (no API call, no cost).
4. Embeddings go into an in-memory FAISS index using cosine similarity.
5. At query time, the question is embedded, the top-5 chunks are retrieved, and they're sent to Claude (`anthropic/claude-sonnet-4` via OpenRouter) with a system prompt that **requires** the answer to end with a `Sources: page X, page Y` line. The model can't dodge the citation.

## Tech stack

- Python 3.10+
- `pypdf` for PDF text extraction
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local embeddings
- `faiss-cpu` for vector retrieval
- `openai` SDK pointed at `https://openrouter.ai/api/v1` for the Claude call
- `python-dotenv` for `.env` loading

No external vector DB. No deployment. No second API key beyond OpenRouter.

## Limitations

Written candidly so a recruiter doesn't have to guess:

- **Single-document only.** This tool answers questions about one PDF at a time. No batching, no per-client document libraries, no cross-document comparison.
- **No PHI handling.** The bundled sample is the publicly available CMS federal SBC template (no member data, no carrier-proprietary content). For real-world use, you would need to add proper data-handling, access controls, and a PHI-safe deployment path.
- **No retrieval cache.** Every question re-embeds the query and hits Claude. Fine for a demo; you'd add caching for real volume.
- **Page-cite trust depends on retrieval.** The system prompt forbids made-up citations, but if retrieval surfaces the wrong chunks, the answer (and the citation) can still be wrong. Always spot-check critical answers against the actual page.
- **Scanned-only PDFs won't work.** No OCR. If the document is image-only, you'd need to run it through Tesseract or similar first.
- **Not deployed.** No public URL. This is code-only — clone, set a key, run.

## Acknowledgements

The bundled sample SBC is the [CMS English Sample Completed SBC (accessible format, 01-28-25)](https://www.cms.gov/cciio/resources/forms-reports-and-other-resources/downloads/english-sample-completed-sbc-accessible-format-012825.pdf) — a public-domain federal government document.

## License

MIT. See `LICENSE`.
