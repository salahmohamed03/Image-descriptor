"""
Message types and structures for PubSub communication.
This module defines all message topics and their expected payload structures
to ensure consistent communication across the application.
"""

class Topics:
    """
    Defines all PubSub topics used for application communication.
    These constants should be used when publishing or subscribing to events.
    """
    # Corner detection topics
    APPLY_CORNER_DETECTION = "apply_corner_detection"
    CORNER_DETECTION_COMPLETE = "corner_detection_complete"
    
    # Matching topics
    APPLY_MATCHING = "apply_matching"
    MATCHING_COMPLETE = "matching_complete"
    
    # SIFT topics
    APPLY_SIFT = "apply_sift"
    SIFT_COMPLETE = "sift_complete"


"""
Message Structures Documentation

Documents the expected payload structure for each topic.
This serves as documentation for developers to understand what data
should be included when publishing messages or what to expect when subscribing.

Corner Detection:
---------------
APPLY_CORNER_DETECTION:
    image: numpy.ndarray - Image to process
    method: str - Detection method (Harris, Lambda-)
    threshold: float - Detection threshold value

CORNER_DETECTION_COMPLETE:
    result_image: numpy.ndarray - Processed image with corners highlighted
    computation_time: float - Processing time in seconds

Matching:
-------
APPLY_MATCHING:
    image1: numpy.ndarray - First image to match
    image2: numpy.ndarray - Second image to match (template)
    method: str - Matching method (NCC, SSD)

MATCHING_COMPLETE:
    result_image: numpy.ndarray - Processed image with match highlighted
    computation_time: float - Processing time in seconds

SIFT:
----
APPLY_SIFT:
    image: numpy.ndarray -  image for SIFT feature generation

SIFT_COMPLETE:
    result_image: numpy.ndarray - Processed image with SIFT matches
    computation_time: float - Processing time in seconds
"""
