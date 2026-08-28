"""Resume compatibility checks specific to static multiplier mappings."""


def validate_static_multiplier_resume(checkpoint, current_configuration):
    """Require the checkpoint and current resolved mappings to be identical."""

    try:
        saved_configuration = checkpoint["run_metadata"]["optimizer"][
            "settings"
        ]["static_multiplier_configuration"]
        saved_mapping = saved_configuration["resolved_multipliers"]
    except KeyError as exc:
        raise ValueError(
            "checkpoint lacks the resolved static multiplier mapping"
        ) from exc

    current_mapping = current_configuration.get("resolved_multipliers")
    if saved_mapping != current_mapping:
        saved_names = set(saved_mapping)
        current_names = set(current_mapping or {})
        changed_names = sorted(
            name
            for name in saved_names & current_names
            if saved_mapping[name] != current_mapping[name]
        )
        missing_names = sorted(saved_names - current_names)
        added_names = sorted(current_names - saved_names)
        details = []
        if changed_names:
            details.append("changed: " + ", ".join(changed_names))
        if missing_names:
            details.append("missing: " + ", ".join(missing_names))
        if added_names:
            details.append("added: " + ", ".join(added_names))
        raise ValueError(
            "checkpoint static multiplier mapping mismatch"
            + (" (" + "; ".join(details) + ")" if details else "")
        )
