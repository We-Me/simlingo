# SimLingo instruction verification

This directory runs closed-loop instruction-following checks with a visible
CARLA window and a separate Pygame window. The default preset runs one Town12
route for each of the three required basic-track scenes.

```bash
# Three-scene smoke test (default)
bash verify_instruction/run_verify_instruction.sh

# Show all 30 instructions and user translations
bash verify_instruction/run_verify_instruction.sh --list

# Run one instruction on its recommended route
bash verify_instruction/run_verify_instruction.sh --instruction-id S1-03

# Chain the five core Scene 1 commands on one custom Town12 route
bash verify_instruction/run_verify_instruction.sh --scene1-sequence

# Run all core competition instructions
bash verify_instruction/run_verify_instruction.sh --preset core

# Run the complete 30-instruction suite
bash verify_instruction/run_verify_instruction.sh --preset all

# Add Town13 emergency cases
bash verify_instruction/run_verify_instruction.sh --preset town13-emergency
```

The model receives only the English `instruction` from `instructions_en.json`.
Chinese text in `instructions_zh.json` is printed for the user and is drawn in
Pygame only when a CJK font is available. Set `SIMLINGO_CJK_FONT` if automatic
font discovery fails.

The agent keeps `GlobalConfig.use_cot=True` and uses the trained
`<INSTRUCTION_FOLLOWING>` language-to-action path. Its instruction-following
language answer is usually the short trained phrase “Following the given
instruction. Waypoints:”; it should be described as language conditioning,
not as a detailed free-form reasoning trace.

Each run writes `case.json`, evaluator results, live status, and visualization
artifacts under `eval_results/verify_instruction/`. Route bindings marked
`partial` do not fully instantiate every subtask in a compound command and
must not be counted as a complete competition-scene pass.

`--scene1-sequence` is separate from the ordinary presets. It runs one
continuous, 2.2 km clear-day Town12 route and changes the active English
command in this order: keep lane, accelerate to 60 km/h, turn right, change
one lane to the left, then slow to 40 km/h. Moving background traffic,
parked-vehicle spawning, and active Bench2Drive scenarios are disabled for
this isolated basic-command check. Phase changes are printed in the terminal,
shown in Pygame, and saved to
`sequence_transitions.jsonl`. The XML contains one off-map metadata anchor
because the released evaluator assumes `scenario_configs[0]` exists; the
route filter removes that anchor before any scenario actor or behavior is
created.

This mode is a visual/behavioral verification run, not a new automatic score.
The transition log records when route progress, the RIGHT navigation command,
and the actual lane ID change are observed. If no left lane change is observed
within its 250 m window, the log records `not_observed` and continues to the
final slowdown phase so the remaining command can still be tested. Judge lane
keeping and the two requested speeds from the Pygame telemetry, saved
visualization, and evaluator output.

Bench2Drive220 has no exact bus-stop scenario. S2-03 and S2-04 therefore use
Town13 route 3248 (`ParkingCrossingPedestrian`) as a functional equivalent:
slow down at a curbside occlusion, watch for pedestrians, and continue only
after the road is clear.
