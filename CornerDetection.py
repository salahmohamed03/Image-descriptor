
from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time


class CornerDetection:
    def __init__(self):
        pass
    def setup_subscriptions(self):
        pub.subscribe(self.on_ApplyCornerDetection, Topics.APPLY_CORNER_DETECTION)

    def on_ApplyCornerDetection(self, image, method, threshold):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image, method, threshold)
    
    def apply(self, image, method, threshold):
        start = time.time()

        # write the main code here

        result_image = None
        ###

        computation_time = time.time() - start
        pub.sendMessage(
            Topics.CORNER_DETECTION_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )