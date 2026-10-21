# Literature check of 21 October 2026: dated rerun with stored responses

The checks of 18–19 and 20 October did not store the API responses (see
[literature-check-2026-10-20.md](literature-check-2026-10-20.md)). This run repeats the API part of the 20 October
check with [scripts/literature_check.py](../scripts/literature_check.py) and stores what came back in
[literature-runs/2026-10-21/](literature-runs/2026-10-21/): for every query the URL, the UTC time, the HTTP status
and the records (identifiers, titles, authors, dates; abstracts are left out on purpose). Rerunning the script
writes a new dated folder that can be compared with this one. A search that finds nothing is evidence, not proof,
that a result is new.

## What came back

| Source | Queries | Answered | Result |
|---|---|---|---|
| arXiv API | the 25 queries of 20 October, newest first, up to 60 records each | 25 of 25 (HTTP 200) | 73 distinct records, 18 of them from 2025–2026. Relevant: only arXiv:2503.10138 (Barzilai, Shamir, Zamani) and arXiv:2509.08954 (Hieu); the other 16 are about unrelated "curves" (3D printing, geodesics, beam shaping, scaling laws, …). |
| Semantic Scholar Graph API | the paper, its citations, 5 keyword searches | 3 of 7; 4 refused by the rate limit (HTTP 429) after 6 tries with growing pauses | Citations: only Hieu (2025). The two searches that were answered find nothing beyond Hieu's paper, which one of them lists (the other titles are about minimax problems, curve searches for heavy-ball methods and convergence rates). The paper record (citation count) and three searches were not answered; they are marked 429 in the file. |
| OpenAlex | the paper (by its arXiv DOI), then `cites:` filter | 2 of 2 | W6966804991, cited_by_count 0; no citing works. |
| OpenReview | the notes of the TMLR forum TZtpxselK2 | 0 of 1 | The API answered HTTP 403. The forum page, opened once in a browser the same day, asked for a browser verification, which was not bypassed. Its reviews and discussion are still not read. |

Google Scholar has no API and is not part of the script; its result of 20 October ("Cited by 3": Hieu, the GPT-5
report, Nagda–Raghavan–Thakurta) is recorded in the log of that day.

## Outcome

Unchanged from 20 October: no indexed work states a threshold for consecutive violations, the threshold h₁(κ) for
strongly convex functions, or the worst-case size of a violation. Not covered: the OpenReview discussion (browser
check), the four Semantic Scholar calls refused by the rate limit, and Google Scholar on this date.
