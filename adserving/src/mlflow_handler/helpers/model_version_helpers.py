# python
from typing import Any, Optional


def create_model_version_from_run(
    client,
    logger,
    model_name: str,
    run_id: str,
    original_source: str,
    comment: str,
) -> Optional[Any]:
    """Create model version using original source but new run."""
    try:
        new_version = client.create_model_version(
            name=model_name,
            source=original_source,  # Original model artifacts
            run_id=run_id,  # New run with updated parameters
            description=f"Parameter update: {comment or 'Updated parameters'}",
        )

        logger.info(
            f"Created v{new_version.version} from run "
            f"{run_id} using source {original_source}"
        )
        return new_version

    except Exception as e:
        logger.error(f"Error creating model version: {e}")
        return None


def transition_to_production_with_archive(
    client, logger, model_name: str, new_version: str
) -> bool:
    """Transition new version to Production."""
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )

        logger.info(f"Transitioned {model_name} v{new_version} to Production")
        return True

    except Exception as e:
        logger.error(f"Error transitioning to Production: {e}")
        return False
