# Baseline: v3.4-advanced-extended

`meta_gen_v3.4.py` is the implementation v3.5 replaced, kept verbatim so the
change can be diffed and the recorded v3.4 output explained. Do not run it — it
is here as a reference only.

It is the file that produced the 368 rows in `course_metadata_regenerated` with
`llm_prompt_version = 'v3.4-advanced-extended'`. Those rows came from three CSV
feeds (`CSV:HMM`, `CSV:Coursera`, `CSV:eCornell`), not from course folders, which
is why every one of them reports no transcript and no PDF text.

Line references for the defects the repository README describes:

| Defect | Line |
|---|---|
| PDFs discovered with `glob` (top level), so none found in the nested layout | 857 |
| PDF text extracted with no character cap | 592 |
| Native PDF passing present but commented out | 684, 691 |
| `explain` sub-objects declared `{"type": "object"}` with no `properties`, so structured output returns `{}` | 527, 532-542 |
| Rubric weights LO5/PK5/BT25/CC35/ERO10/EOL20/TAA0 | 362-363 |
| Classification bands 0-55 / 56-75 / 76-100 | 365-367 |
| `del current_metadata['competencies_v6']` — KeyError discards the whole record | 841 |
| SCORM read once from an env var rather than per course | 36, 1008 |
| Transcript read only at the course root | 846-847 |
| Retry on every exception; no request timeout | 667 |
| `prompt.txt` written on every call, raced by concurrent workers | 700-704 |
| Reference paths that do not exist (`data/competencies.json`, `data/SGOS.json`) | 34-35 |
| Sector/SubSector/SubSectorTheme required while the prompt forbids populating them | 548 |
| Output keyed on `course_id` alone, so a re-run overwrites the prior generation | 777, 786 |
