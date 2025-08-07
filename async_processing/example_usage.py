#!/usr/bin/env python3
"""
Example usage script for the Async Processing System

This script demonstrates how to:
1. Initialize the async processing system
2. Send training data and inference results
3. Start workers for processing
4. Monitor system status
5. Retrieve performance statistics

Prerequisites:
- Redis server running on localhost:6379
- PostgreSQL server running on localhost:5432
- Required Python packages installed (see requirements.txt)

Usage:
    python example_usage.py
"""

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the async processing system
try:
    from . import AsyncProcessingConfig, AsyncProcessingSystem
    from .config import load_async_config_from_env
except ImportError:
    # For running as standalone script
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from async_processing import AsyncProcessingConfig, AsyncProcessingSystem
    from async_processing.config import load_async_config_from_env


async def setup_system() -> AsyncProcessingSystem:
    """Initialize and setup the async processing system"""
    logger.info("Setting up async processing system...")

    # Load configuration from environment
    config = load_async_config_from_env()

    # Create system instance
    system = AsyncProcessingSystem(config)

    # Initialize all components
    await system.initialize()

    # Setup database schema
    try:
        await system.setup_database()
        logger.info("Database schema setup completed")
    except Exception as e:
        logger.warning(f"Database schema setup failed: {e}")
        logger.info("Continuing without database setup...")

    return system


async def generate_sample_data() -> Dict[str, Any]:
    """Generate sample training data and inference results"""
    models = ["fraud_detection", "recommendation_engine", "price_optimizer"]
    model_name = random.choice(models)
    request_id = str(uuid.uuid4())

    # Sample input data
    input_data = {
        "user_id": random.randint(1000, 9999),
        "transaction_amount": round(random.uniform(10.0, 1000.0), 2),
        "merchant_category": random.choice(["grocery", "gas", "restaurant", "online"]),
        "time_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "features": [random.uniform(-1, 1) for _ in range(10)],
    }

    # Sample prediction result
    prediction = {
        "prediction": random.choice([0, 1]),
        "probability": round(random.uniform(0.1, 0.9), 3),
        "model_version": "v1.2.3",
    }

    # Additional metadata
    metadata = {
        "api_version": "v1",
        "processing_time": random.randint(50, 200),
        "model_features_used": len(input_data["features"]),
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "model_name": model_name,
        "request_id": request_id,
        "input_data": input_data,
        "prediction": prediction,
        "metadata": metadata,
        "confidence": prediction["probability"],
        "anomaly_score": random.uniform(0.0, 1.0),
        "is_anomaly": prediction["probability"] < 0.3,
        "processing_time": metadata["processing_time"],
        "source_ip": f"192.168.1.{random.randint(1, 254)}",
        "user_agent": "AsyncProcessingExample/1.0",
    }


async def send_sample_data(system: AsyncProcessingSystem, num_samples: int = 10):
    """Send sample data to the async processing system"""
    logger.info(f"Sending {num_samples} sample data records...")

    for i in range(num_samples):
        try:
            # Generate sample data
            data = await generate_sample_data()

            # Send complete API request data
            message_ids = await system.send_api_request_data(
                model_name=data["model_name"],
                request_id=data["request_id"],
                input_data=data["input_data"],
                prediction=data["prediction"],
                metadata=data["metadata"],
                confidence=data["confidence"],
                anomaly_score=data["anomaly_score"],
                is_anomaly=data["is_anomaly"],
                processing_time=data["processing_time"],
                source_ip=data["source_ip"],
                user_agent=data["user_agent"],
            )

            logger.info(f"Sent sample {i+1}: {data['model_name']} - {message_ids}")

            # Small delay between sends
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to send sample {i+1}: {e}")

    logger.info("Finished sending sample data")


async def monitor_system_status(system: AsyncProcessingSystem):
    """Monitor and display system status"""
    logger.info("Checking system status...")

    try:
        status = await system.get_system_status()

        logger.info("=== System Status ===")
        logger.info(f"Initialized: {status.get('initialized', False)}")

        # Configuration status
        config = status.get("config", {})
        logger.info(
            f"Async Processing Enabled: {config.get('async_processing_enabled', False)}"
        )
        logger.info(f"Number of Workers: {config.get('num_workers', 0)}")
        logger.info(f"Stream Names: {config.get('stream_names', {})}")

        # Producer status
        producer = status.get("producer", {})
        if producer:
            logger.info(f"Producer Health: {producer.get('health', False)}")
            for stream_name, stream_info in producer.items():
                if isinstance(stream_info, dict) and "length" in stream_info:
                    logger.info(
                        f"Stream {stream_name}: {stream_info.get('length', 0)} messages"
                    )

        # Database status
        database = status.get("database", {})
        logger.info(f"Database Health: {database.get('health', False)}")

        # Workers status
        workers = status.get("workers", [])
        healthy_workers = sum(1 for w in workers if w.get("overall_health", False))
        logger.info(f"Healthy Workers: {healthy_workers}/{len(workers)}")

        return status

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {}


async def get_performance_stats(system: AsyncProcessingSystem):
    """Get and display model performance statistics"""
    logger.info("Retrieving performance statistics...")

    models = ["fraud_detection", "recommendation_engine", "price_optimizer"]

    for model_name in models:
        try:
            stats = await system.get_model_performance_stats(model_name, hours=1)

            if stats and stats.get("total_requests", 0) > 0:
                logger.info(f"=== {model_name} Performance (Last Hour) ===")
                logger.info(f"Total Requests: {stats.get('total_requests', 0)}")
                logger.info(
                    f"Success Rate: {stats.get('success_count', 0)}/{stats.get('total_requests', 0)}"
                )
                logger.info(f"Average Confidence: {stats.get('avg_confidence', 0):.3f}")
                logger.info(
                    f"Average Anomaly Score: {stats.get('avg_anomaly_score', 0):.3f}"
                )
                logger.info(f"Anomaly Count: {stats.get('anomaly_count', 0)}")
                logger.info(
                    f"Average Processing Time: {stats.get('avg_processing_time', 0):.1f}ms"
                )
            else:
                logger.info(f"No data found for model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to get stats for {model_name}: {e}")


async def run_workers_demo(system: AsyncProcessingSystem, duration_seconds: int = 30):
    """Run workers for a specified duration to process messages"""
    logger.info(f"Starting workers for {duration_seconds} seconds...")

    try:
        # Start workers in background
        worker_task = asyncio.create_task(system.start_workers())

        # Wait for specified duration
        await asyncio.sleep(duration_seconds)

        # Stop workers
        await system.stop_workers()

        # Cancel the worker task
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        logger.info("Workers stopped")

    except Exception as e:
        logger.error(f"Error running workers: {e}")


async def main():
    """Main demonstration function"""
    logger.info("Starting Async Processing System Demo")

    system = None
    try:
        # 1. Setup system
        system = await setup_system()
        logger.info("✓ System initialized successfully")

        # 2. Check initial status
        await monitor_system_status(system)

        # 3. Send sample data
        await send_sample_data(system, num_samples=20)
        logger.info("✓ Sample data sent")

        # 4. Run workers to process the data
        await run_workers_demo(system, duration_seconds=10)
        logger.info("✓ Workers processed data")

        # 5. Check final status
        await monitor_system_status(system)

        # 6. Get performance statistics
        await get_performance_stats(system)

        logger.info("✓ Demo completed successfully")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if system:
            try:
                await system.close()
                logger.info("✓ System closed cleanly")
            except Exception as e:
                logger.error(f"Error closing system: {e}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
