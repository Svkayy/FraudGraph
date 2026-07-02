from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable

from . import config


def ensure_topics() -> bool:
    """Create the input/output topics if absent. Returns False if no broker
    is reachable (serving can still run in HTTP-only mode)."""
    try:
        admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP)
    except NoBrokersAvailable:
        return False
    topics = [
        NewTopic(config.TOPIC_IN, num_partitions=config.NUM_PARTITIONS, replication_factor=1),
        NewTopic(config.TOPIC_OUT, num_partitions=config.NUM_PARTITIONS, replication_factor=1),
    ]
    for t in topics:
        try:
            admin.create_topics([t])
        except TopicAlreadyExistsError:
            pass
    admin.close()
    return True
