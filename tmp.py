from __future__ import print_function
import time
import uuid
import sys
import socket
import elasticache_auto_discovery
from pymemcache.client.hash import HashClient

print("Imported")
# elasticache settings
elasticache_config_endpoint = "cache-test-mem.sejiby.cfg.use1.cache.amazonaws.com:11211"
nodes = elasticache_auto_discovery.discover(elasticache_config_endpoint)
print("P1")
nodes = map(lambda x: (x[1], int(x[2])), nodes)
memcache_client = HashClient(nodes)
print("Connected")


def lambda_handler(event, context):
    """
    This function puts into memcache and get from it.
    Memcache is hosted using elasticache
    """

    test_round = 500
    start_t = time.time()
    # Create a random UUID... this will be the sample element we add to the cache.
    uuid_inserted = uuid.uuid4().hex
    # Put the UUID to the cache.
    for i in range(test_round):
        memcache_client.set('uuid', uuid_inserted)
    set_time = time.time() - start_t
    print("Avg set time: {} ms".format(set_time * 1000.0 / test_round))

    # Get item (UUID) from the cache.
    start_t = time.time()
    for i in range(test_round):
        uuid_obtained = memcache_client.get('uuid')
    set_time = time.time() - start_t
    print("Avg get time: {} ms".format(set_time * 1000.0 / test_round))

    if uuid_obtained.decode("utf-8") == uuid_inserted:
        # this print should go to the CloudWatch Logs and Lambda console.
        print("Success: Fetched value %s from memcache" % (uuid_inserted))
    else:
        raise Exception("Value is not the same as we put :(. Expected %s got %s" % (uuid_inserted, uuid_obtained))

    return "Fetched value from memcache: " + uuid_obtained.decode("utf-8")