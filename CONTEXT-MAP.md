src/wilcus_vault/qualify.py:81 — `confined_path` sits before the per-linker `try`; move it inside
src/wilcus_vault/paths.py:22 — `confined_path`; raises "passes through a symlink" for a stale linker row
src/wilcus_vault/indexer.py:180-199 — `index_paths` already carries `index_error` into `IndexStats`; file is AT the 200-line cap
src/wilcus_vault/doctor.py:33-85,173-186 — `DoctorReport` (add `index_error` last), line 70 drops it, `_rebuild_index` drops the result; 186 lines
src/wilcus_vault/cli/usage.py:76-77,81-102 — `summary` already prints `index_error`; `print_report` must too; exit-code usage text
src/wilcus_vault/cli/__init__.py:95-98 — doctor exit code must include `index_error`
tests/test_indexer_qualify_skips.py — qualify skip tests; add the parent-dir-becomes-symlink case
tests/test_doctor.py — doctor report tests; add index_error on repair and --rebuild
tests/test_doctor_embed.py — precedent for doctor embed-failure tests
tests/test_cli.py:43,60 — doctor exit-code assertions to extend
tests/test_lines.py — 200-line cap on src/wilcus_vault
