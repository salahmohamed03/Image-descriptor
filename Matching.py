from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import logging
import time


class MatchingProcessor:
    def __init__(self):
        self.setup_subscriptions()
        logging.info("Matching processor initialized")
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_matching, Topics.APPLY_MATCHING)
    
    def on_apply_matching(self, image1, image2, method):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image1, image2, method)
    
    def apply(self, image1, image2, method):
        start = time.time()

        # write the main code here

        result_image = None
        ###

        computation_time = time.time() - start
        
        # Send the result back through PubSub
        pub.sendMessage(
            Topics.MATCHING_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )