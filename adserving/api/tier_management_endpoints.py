"""
Tier Management Endpoints for Business-Critical Models
"""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .api_dependencies import (
    get_tier_orchestrator,
    is_tier_based_deployment_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_tier_deployment_enabled():
    """Check if tier-based deployment is enabled"""
    if not is_tier_based_deployment_enabled():
        raise HTTPException(
            status_code=400, detail="Tier-based deployment is not enabled"
        )


@router.post("/tier-management/manual-assignment")
async def set_manual_tier_assignment(
    model_name: str, tier: str, is_business_critical: bool = True
):
    """Set manual tier assignment for a model"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        success = orchestrator.tier_manager.set_manual_tier_assignment(
            model_name, tier, is_business_critical
        )

        if success:
            return {
                "status": "success",
                "message": f"Manual tier assignment set: {model_name} -> {tier}",
                "model_name": model_name,
                "tier": tier,
                "is_business_critical": is_business_critical,
            }
        else:
            raise HTTPException(
                status_code=400, detail="Failed to set manual tier assignment"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting manual tier assignment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set manual tier assignment: {str(e)}"
        )


@router.delete("/tier-management/manual-assignment/{model_name}")
async def remove_manual_tier_assignment(model_name: str):
    """Remove manual tier assignment for a model"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        success = orchestrator.tier_manager.remove_manual_tier_assignment(model_name)

        if success:
            return {
                "status": "success",
                "message": f"Manual tier assignment removed for {model_name}",
                "model_name": model_name,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No manual tier assignment found for {model_name}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing manual tier assignment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove manual tier assignment: {str(e)}"
        )


@router.get("/tier-management/manual-assignment/{model_name}")
async def get_manual_tier_assignment(model_name: str):
    """Get manual tier assignment for a model"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        assignment = orchestrator.tier_manager.get_manual_tier_assignment(model_name)
        current_tier = orchestrator.tier_manager.get_model_tier(model_name)
        is_business_critical = orchestrator.tier_manager.is_business_critical(
            model_name
        )

        return {
            "model_name": model_name,
            "manual_assignment": assignment,
            "current_tier": current_tier,
            "is_business_critical": is_business_critical,
            "is_manually_assigned": assignment is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting manual tier assignment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get manual tier assignment: {str(e)}"
        )


@router.get("/tier-management/business-critical-models")
async def get_business_critical_models():
    """Get all business-critical models and their assignments"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        business_critical_models = (
            orchestrator.tier_manager.get_business_critical_models()
        )

        return {
            "business_critical_models": business_critical_models,
            "total_count": len(business_critical_models),
            "patterns": orchestrator.tier_manager.manual_assignment_patterns,
            "default_tier": (orchestrator.tier_manager.default_business_critical_tier),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting business-critical models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get business-critical models: {str(e)}"
        )


@router.post("/tier-management/business-critical-pattern")
async def add_business_critical_pattern(pattern: str, tier: str = None):
    """Add a pattern for automatic business-critical model detection"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()

        # Add pattern
        orchestrator.tier_manager.add_business_critical_pattern(pattern)

        # Set tier mapping if provided
        if tier and tier in ["hot", "warm", "cold"]:
            orchestrator.tier_manager.pattern_tier_assignments[pattern] = tier

        return {
            "status": "success",
            "message": f"Business-critical pattern added: {pattern}",
            "pattern": pattern,
            "tier": (tier or orchestrator.tier_manager.default_business_critical_tier),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding business-critical pattern: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add business-critical pattern: {str(e)}"
        )


@router.delete("/tier-management/business-critical-pattern")
async def remove_business_critical_pattern(pattern: str):
    """Remove a business-critical pattern"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        success = orchestrator.tier_manager.remove_business_critical_pattern(pattern)

        # Also remove from pattern tier assignments
        if pattern in orchestrator.tier_manager.pattern_tier_assignments:
            del orchestrator.tier_manager.pattern_tier_assignments[pattern]

        if success:
            return {
                "status": "success",
                "message": f"Business-critical pattern removed: {pattern}",
                "pattern": pattern,
            }
        else:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing business-critical pattern: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove business-critical pattern: {str(e)}",
        )


@router.post("/tier-management/bulk-assignment")
async def bulk_assign_business_critical_models(model_assignments: Dict[str, str]):
    """Bulk assign multiple models as business-critical"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        results = orchestrator.tier_manager.bulk_assign_business_critical_models(
            model_assignments
        )

        successful_assignments = {k: v for k, v in results.items() if v}
        failed_assignments = {k: v for k, v in results.items() if not v}

        # Check for conflicts
        if len(failed_assignments) > 0 and len(successful_assignments) > 0:

            return JSONResponse(
                status_code=409,
                content={
                    "status": "partial_success",
                    "successful_assignments": successful_assignments,
                    "failed_assignments": failed_assignments,
                    "total_requested": len(model_assignments),
                    "successful_count": len(successful_assignments),
                    "failed_count": len(failed_assignments),
                    "message": "Bulk assignment completed with some failures",
                },
            )

        return {
            "status": "completed",
            "successful_assignments": successful_assignments,
            "failed_assignments": failed_assignments,
            "total_requested": len(model_assignments),
            "successful_count": len(successful_assignments),
            "failed_count": len(failed_assignments),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk assignment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform bulk assignment: {str(e)}"
        )


@router.get("/tier-management/statistics")
async def get_tier_management_statistics():
    """Get comprehensive tier management statistics"""
    try:
        _check_tier_deployment_enabled()

        orchestrator = get_tier_orchestrator()
        tier_stats = orchestrator.get_tier_statistics()

        # Add manual assignment statistics
        manual_assignments_count = len(
            orchestrator.tier_manager.manual_tier_assignments
        )
        business_critical_count = len(
            orchestrator.tier_manager.business_critical_models
        )
        patterns_count = len(orchestrator.tier_manager.manual_assignment_patterns)

        tier_stats["manual_assignments"] = {
            "total_manual_assignments": manual_assignments_count,
            "business_critical_models": business_critical_count,
            "active_patterns": patterns_count,
            "prevent_automatic_changes": (
                orchestrator.tier_manager.prevent_automatic_changes
            ),
            "default_business_critical_tier": (
                orchestrator.tier_manager.default_business_critical_tier
            ),
        }

        return tier_stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tier management statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tier management statistics: {str(e)}",
        )
