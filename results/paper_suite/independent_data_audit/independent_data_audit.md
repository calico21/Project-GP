# Independent-data eligibility audit

The repository does not contain an independent target for all 108 production-model states. The CAN files are eligible only for directly logged, session-held-out observables; source-file hashes and channel availability are recorded in the accompanying JSON. The TTC archive has data from three runs, but its existing `is_test` flag partitions rows within every run. It must not support a run-generalisation claim. Use a complete-run holdout and training-only scaling for a tire submodel study.
