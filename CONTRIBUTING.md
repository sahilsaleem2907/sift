# Contributing to Sift

Thanks for helping make Sift better. Sift is open source under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, and it stays that way because of contributions like yours.

## Ways to contribute

- **Improve the reviews.** Add a new check, tune routing, sharpen a prompt, teach Sift a new language — then open a PR.
- **Fix a bug.** Spot something wrong? Send a fix.
- **Raise an issue.** Found a bug or have an idea but don't want to write the code? [Open an issue](https://github.com/sahilsaleem2907/sift/issues).

## Development setup

Sift is a self-hostable backend. To run it locally you need:

- **PostgreSQL** (connection string via `DATABASE_URL`)
- An **LLM reachable by LiteLLM** — bring your own key (BYOK). See [.env.example](.env.example) for OpenAI, Anthropic, Gemini, Azure, Bedrock, or local Ollama.

```bash
cp .env.example .env      # then fill in DATABASE_URL and your LLM settings
# run via Docker (see README) or your local Python environment
```

Please run the linters and tests before opening a PR:

```bash
ruff check .
# run the test suite as configured in the repo
```

## Pull request flow

1. Fork the repo and create a branch off `dev`.
2. Make your change. Keep it focused — one concern per PR.
3. Add or update tests where it makes sense.
4. Ensure `ruff` is clean and existing tests pass.
5. **Sign off your commits** (see below).
6. Open a PR against `dev` describing the change and the motivation.

## Signing off your commits (Developer Certificate of Origin)

Sift uses the [Developer Certificate of Origin (DCO)](https://developercertificate.org/) rather than a CLA. The DCO is a simple statement that you wrote the contribution, or otherwise have the right to submit it under the project's license.

To agree, add a `Signed-off-by` line to each commit — Git does this for you with the `-s` flag:

```bash
git commit -s -m "Add my improvement"
```

This appends a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and an email you can be reached at. Commits without a sign-off cannot be merged.

Contributions are accepted under AGPL-3.0 — the same license as the project (inbound = outbound). By signing off, you license your contribution to the project and its users under AGPL-3.0.

## License

By contributing, you agree that your contributions are licensed under the [AGPL-3.0](LICENSE), consistent with the DCO sign-off above.
