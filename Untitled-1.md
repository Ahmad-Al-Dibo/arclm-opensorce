``` 
# PROMPT 5 — Documentation, Examples & Website as the Final Product Surface

The ArcLM architecture has now been cleaned and the core APIs have been migrated.

Only now update the documentation and website.

Do not use old documentation as truth simply because it exists.

Treat the current working code and tested public architecture as the source of truth.

First audit all documentation and remove or rewrite references to:

- removed APIs
- old Trainer architecture
- obsolete tokenizer classes
- outdated vNext migration concepts
- removed compatibility layers
- old examples
- incorrect model-loading behavior
- outdated fine-tuning instructions

The documentation should clearly explain the three ArcLM usage levels:

1. Student / Lab
2. Professional
3. Research

Create a clear getting-started path that allows a new user to understand ArcLM without reading internal architecture documentation.

Documentation should include working examples for:

- installation
- inspecting data
- loading/creating a model
- tokenizer usage
- training
- fine-tuning
- validation
- saving `.arcmodel`
- loading `.arcmodel`
- professional configuration
- research customization

All code examples must be tested or derived from tested examples.

Do not publish fake capabilities.

If something is planned but not implemented, explicitly mark it as planned rather than showing it as working.

## Website

Now redesign/clean the ArcLM website so it represents the actual framework rather than an experimental project page.

The website should communicate:

- what ArcLM is
- why it exists
- who it is for
- what is currently supported
- how to install it
- how to start
- Student / Professional / Research workflows
- training
- fine-tuning
- model/tokenizer architecture
- `.arcmodel`
- API/documentation navigation
- GitHub/project links where already available

The website should feel like a real open-source developer framework website.

Use clean technical design rather than marketing exaggeration.

Every technical claim on the website must correspond to functionality currently present in the repository.

Remove stale pages, duplicate content and outdated architecture descriptions.

Do not invent benchmarks, adoption statistics, supported models or performance numbers.

Where useful, generate documentation automatically from stable public APIs rather than manually duplicating signatures everywhere.

Finally run:

- code tests
- documentation examples
- import tests
- package/build checks
- website build
- broken-link checks where available

Finish with a final ArcLM vNext readiness report containing:

- architecture status
- Student API status
- Professional API status
- Research API status
- model independence status
- Transformers dependency status
- tokenizer status
- training status
- fine-tuning status
- `.arcmodel` status
- tests
- documentation status
- website status
- remaining technical debt
- recommended next major milestone

The result should be a repository that looks and behaves like one coherent professional framework, not a collection of old and new experiments.