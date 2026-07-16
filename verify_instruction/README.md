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

Bench2Drive220 has no exact bus-stop scenario. S2-03 and S2-04 therefore use
Town13 route 3248 (`ParkingCrossingPedestrian`) as a functional equivalent:
slow down at a curbside occlusion, watch for pedestrians, and continue only
after the road is clear.
