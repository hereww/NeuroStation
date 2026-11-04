"""Version 2 trial ledger. Legacy FFT scores are not model acceptances."""
from collections import defaultdict


def ratio(numerator, denominator, population):
    return {"numerator": numerator, "denominator": denominator, "population": population,
            "value": numerator / denominator if denominator else None}


def summarize_trials(ledger):
    counts = {"all": len(ledger)}
    for key, field, value in (("quality_eligible", "quality", "pass"),
                              ("model_accepted", "decision", "accepted"),
                              ("rejected", "decision", "rejected"),
                              ("unassessable", "decision", "unassessable")):
        counts[key] = sum(row.get(field) == value for row in ledger)
    rates = {"acceptance": ratio(counts["model_accepted"], counts["all"], "all recorded trials")}
    for label in ("presented_target_id", "gaze_target_id"):
        eligible = [r for r in ledger if r.get("quality") == "pass" and r.get(label)
                    and r.get("predicted_target_id") and r.get("scoring_status") == "scored"]
        rates[label + "_quality_scored_accuracy"] = ratio(
            sum(r[label] == r["predicted_target_id"] for r in eligible), len(eligible),
            "quality-eligible scored trials with this label; not model acceptances")
    return {"schema_version": 2, "counts": counts, "rates": rates,
            "labels": {"presented_target_id": "protocol presentation; not gaze proof",
                       "gaze_target_id": "manual/self report; not eye tracking"},
            "acceptance_authority": "explicit model decision only; FFT winner is not accepted"}


def build_trial_ledger(events, scored, sample_count):
    groups = defaultdict(list)
    for event in events:
        key = str(event.get("trial_index", ""))
        if key not in ("", "None"):
            groups[key].append(event)
    scores = {str(r["trial_index"]): r for r in scored}
    ledger = []
    for key, group in groups.items():
        onsets = [r for r in group if r.get("event_name") == "stimulus_onset"]
        offsets = [r for r in group if r.get("event_name") == "stimulus_offset"]
        score = scores.get(key)
        reason = None
        if any("abort" in str(r.get("event_name", "")) for r in group):
            reason = "aborted"
        elif len(onsets) != 1 or len(offsets) != 1:
            reason = "missing_or_ambiguous_events"
        else:
            try:
                start, stop = int(onsets[0]["sample_index"]), int(offsets[0]["sample_index"])
                if not 0 <= start < stop <= sample_count:
                    reason = "invalid_or_incomplete_interval"
            except (ValueError, TypeError, KeyError):
                reason = "unknown_sample_association"
        if reason is None and score is None:
            reason = "scoring_failed"
        row = {"trial_index": key, "quality": "unknown", "decision": "unassessable",
               "scoring_status": "unassessable", "reason": reason,
               "presented_target_id": (onsets[0].get("presented_target_id") or onsets[0].get("target_id")) if onsets else None,
               "gaze_target_id": None, "predicted_target_id": None}
        if reason is None:
            row.update(quality="fail" if score["artifact_rejected"] else "pass",
                       decision="rejected" if score["artifact_rejected"] else "unassessable",
                       scoring_status="scored", reason="artifact" if score["artifact_rejected"] else "no_model_decision",
                       presented_target_id=score["presented_target_id"], gaze_target_id=score["gaze_target_id"],
                       predicted_target_id=score["eeg_predicted_target_id"])
        ledger.append(row)
    return ledger
