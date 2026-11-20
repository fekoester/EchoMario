#!/usr/bin/env python
"""Behavioural comparison: imitation student vs planner teacher vs constant run+jump."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from echomario.envs.make_env import make_env
from echomario.experts.planner import ACTION_NAMES, LookaheadPlanner, unwrap_toy
from echomario.imitation import FeatureMap, discrete_config

RJ = ACTION_NAMES.index("right_run_jump")
JUMP_ACTIONS = {i for i, n in enumerate(ACTION_NAMES) if "jump" in n}
BACK_OR_WAIT = {ACTION_NAMES.index(n) for n in ("idle", "left", "left_jump", "jump")}


def play(args):
    kind, run_dir, seeds = args
    res = json.loads((Path(run_dir) / "results.json").read_text())
    cfg = res["config"]
    mode = res["mode"]
    W = np.load(Path(run_dir) / "readout.npy")
    env = make_env(discrete_config(cfg))
    fm = FeatureMap(cfg, int(env.observation_space.shape[0]), mode)
    planner = LookaheadPlanner()
    out = []
    for seed in seeds:
        obs, _ = env.reset(options={"level_seed": seed})
        fm.reset()
        planner.attach(env)
        base = unwrap_toy(env)
        bridges = [(p.x0, p.x1, p.y) for p in base.bridge_platforms]
        gaps = [(a, b, any(x0 <= a and x1 >= b for x0, x1, _ in bridges)) for a, b in base.gaps]
        rec = dict(
            seed=seed,
            actions=[],
            regrets=[],
            teacher_actions=[],
            jump_starts=[],
            jump_holds=[],
            on_bridge=0,
            on_raised=0,
            frames=0,
        )
        step, action = 0, RJ
        while True:
            f = fm.step(obs)
            p = base.player
            if kind == "student":
                action = int(np.argmax(f @ W))
                if step % 2 == 0:
                    sc = planner.scores(env)
                    rec["regrets"].append(float(sc.max() - sc[action]))
                    rec["teacher_actions"].append(planner.choose(sc))
            elif kind == "teacher":
                if step % 2 == 0:
                    action = planner.act(env)
            else:
                action = RJ
            if p.on_ground and action in JUMP_ACTIONS:
                rec["jump_starts"].append(round(p.x, 1))
            if p.on_ground and p.y > 1.5:
                rec["on_raised"] += 1
                if any(x0 - 0.5 <= p.x <= x1 + 0.5 and abs(p.y - (y + 1)) < 0.1 for x0, x1, y in bridges):
                    rec["on_bridge"] += 1
            rec["actions"].append(action)
            obs, _r, te, tr, info = env.step(action)
            step += 1
            if te or tr:
                break
        rec["frames"] = step
        rec["finished"] = bool(info["reached_goal"])
        rec["max_x"] = base.max_x
        rec["goal_x"] = base.goal_x
        rec["end"] = dict(x=info["x"], fell=info["fell"], enemy=info["hit_enemy"], stag=info["stagnated"])
        rec["coins"] = int(info["coins_collected"])
        rec["stomps"] = None
        # obstacle outcomes
        obst = []
        for a, b, bridged in gaps:
            if base.max_x < a - 1:
                continue
            died_here = info["fell"] and a - 2 <= info["x"] <= b + 2
            obst.append(
                dict(
                    kind="bridge_gap" if bridged else "gap",
                    width=b - a + 1,
                    cleared=base.max_x > b + 0.5,
                    died=died_here,
                )
            )
        rec["obstacles"] = obst
        # where was the stall / death
        x = info["x"]
        rec["end"]["under_bridge"] = any(x0 - 1 <= x <= x1 + 1 for x0, x1, _ in bridges)
        out.append(rec)
    env.close()
    return kind, run_dir, out


def summarize(kind, recs):
    acts = Counter(a for r in recs for a in r["actions"])
    total = sum(acts.values())
    s = dict(
        kind=kind,
        finished=np.mean([r["finished"] for r in recs]),
        progress=np.mean([r["max_x"] / r["goal_x"] for r in recs]),
    )
    s["action_share"] = {ACTION_NAMES[a]: round(c / total, 3) for a, c in acts.most_common()}
    s["share_right_run_jump"] = round(acts[RJ] / total, 3)
    s["share_wait_or_back"] = round(sum(acts[a] for a in BACK_OR_WAIT) / total, 3)
    tiles = sum(r["max_x"] for r in recs)
    s["jump_starts_per_10_tiles"] = round(10 * sum(len(r["jump_starts"]) for r in recs) / tiles, 2)
    s["frames_on_raised_platform"] = round(sum(r["on_raised"] for r in recs) / sum(r["frames"] for r in recs), 3)
    s["frames_on_bridge"] = round(sum(r["on_bridge"] for r in recs) / sum(r["frames"] for r in recs), 3)
    s["coins_per_level"] = round(np.mean([r["coins"] for r in recs]), 1)
    for k in ("gap", "bridge_gap"):
        ob = [o for r in recs for o in r["obstacles"] if o["kind"] == k]
        if ob:
            s[f"{k}_reached"] = len(ob)
            s[f"{k}_cleared"] = round(np.mean([o["cleared"] for o in ob]), 2)
    ends = Counter()
    for r in recs:
        e = r["end"]
        if r["finished"]:
            ends["finished"] += 1
        elif e["fell"]:
            ends["fell_under_bridge" if e["under_bridge"] else "fell_simple_gap"] += 1
        elif e["enemy"]:
            ends["enemy"] += 1
        elif e["stag"]:
            ends["stalled_under_bridge" if e["under_bridge"] else "stalled_elsewhere"] += 1
        else:
            ends["timeout"] += 1
    s["episode_end"] = dict(ends)
    if recs[0]["regrets"]:
        reg = np.concatenate([r["regrets"] for r in recs])
        ta = [a for r in recs for a in r["teacher_actions"]]
        sa = [r["actions"][i] for r in recs for i in range(0, len(r["actions"]), 2)][: len(ta)]
        s["exact_agreement"] = round(float(np.mean(np.array(sa) == np.array(ta))), 3)
        s["as_good_as_teacher"] = round(float(np.mean(reg < 0.5)), 3)
        s["fatal_choices_per_100_decisions"] = round(float(100 * np.mean(reg >= 100)), 2)
        # when the teacher wants to wait/back off, does the student?
        ta = np.array(ta)
        sa = np.array(sa)
        wait = np.isin(ta, list(BACK_OR_WAIT))
        s["teacher_wait_or_back_share"] = round(float(wait.mean()), 3)
        s["student_also_waits_when_teacher_does"] = (
            round(float(np.isin(sa[wait], list(BACK_OR_WAIT)).mean()), 3) if wait.any() else None
        )
    return s


if __name__ == "__main__":
    stages = {
        "gaps": "runs/imitation/gaps_res_local_multi3000_c5_long",
        "enemies": "runs/imitation/enemies_res_local_multi_c5",
    }
    seeds = list(range(10000, 10020))
    jobs = []
    for run in stages.values():
        for kind in ("student", "teacher", "bunnyhop"):
            for part in (seeds[:10], seeds[10:]):
                jobs.append((kind, run, part))
    with mp.get_context("fork").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        results = pool.map(play, jobs, chunksize=1)
    merged = {}
    for kind, run, recs in results:
        merged.setdefault((run, kind), []).extend(recs)
    out = {}
    for (run, kind), recs in merged.items():
        stage = [k for k, v in stages.items() if v == run][0]
        out[f"{stage}/{kind}"] = summarize(kind, recs)
        print(stage, json.dumps(out[f"{stage}/{kind}"]), flush=True)
    Path("runs/analysis_student.json").write_text(json.dumps(out, indent=2, default=float))
