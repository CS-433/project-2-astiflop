*lifespan_summary.csv* is the summary table containing the following data:
    Filename: worm's uid (e.g. *20240924_piworm09_1*), uid used in the worm tracking data filename (e.g. *coordinates_highestspeed_20240924_09_1_with_time_speed.csv*).
    TimeInfoMergedVideo: Time (in hours.minutes) when the worm stopped moving in the merged video (can be ignored).
    PlateHasDried: TRUE if the agar plate dried by the end of the experiment.
    LifespanInFrames: Number of frames the worm was tracked (1 frame every 2 seconds).
    LifespanInHours: Calculated as LifespanInFrames × 6 ÷ 900.
    Example: 49500 frames = 330 hours.
    Terbinafine: TRUE if treated with terbinafine (a lifespan-extending compound).


Tracking Data (per worm) in folder *TERBINAFINE- (control)* for worms not treated with terbinafine (e.g. FALSE in the Terbinafine column of summary table), or in folder *TERBINAFINE+* for worms treated with terbinafine:
    GlobalFrame: Frame index in the full dataset.
    Timestamp: ISO-formatted real time.
    Speed: Movement speed in pixels/second from t-1 to t.
    Fragment / LocalFrame: Source fragment and local index (can be ignored).
    X, Y: Worm position in pixels.


Recording Setup
    1 frame every 2 seconds
    30-minute sessions (900 frames) every 6 hours
    1 frame = 2 seconds of real time
    Lifespan in hours = frames × 6 ÷ 900

