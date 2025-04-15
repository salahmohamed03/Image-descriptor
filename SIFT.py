from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time


class SIFTProcessor:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_sift, Topics.APPLY_SIFT)
    
    def on_apply_sift(self, image):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image)
    
    def apply(self, image):
        start = time.time()

        # write the main code here

        result_image = None
        ###

        computation_time = time.time() - start
        
        # Send the result back through PubSub
        pub.sendMessage(
            Topics.SIFT_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )
            