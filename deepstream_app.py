#!/usr/bin/env python3
"""
DeepStream Python App Skeleton
- Source → Streammux → nvosd → Sink
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import pyds

# Initialize GStreamer
Gst.init(None)

def bus_call(bus, message, loop):
    """Callback for GStreamer bus messages"""
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("[INFO] End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}: {debug}")
        loop.quit()
    return True

def main(args):
    # Create GStreamer pipeline
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("[ERROR] Unable to create Pipeline\n")

    # --- Source: Filesrc / RTSP ---
    source = Gst.ElementFactory.make("uridecodebin", "source")
    if not source:
        sys.stderr.write("[ERROR] Unable to create Source\n")
    # Pass the video file or RTSP URL here
    uri = args[1]
    source.set_property('uri', uri)

    # --- Stream Muxer ---
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("[ERROR] Unable to create NvStreamMux\n")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    # --- NVVideoConverter ---
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvid-converter")
    if not nvvidconv:
        sys.stderr.write("[ERROR] Unable to create nvvideoconvert\n")

    # --- NVOSD (Drawing Overlay) ---
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    if not nvosd:
        sys.stderr.write("[ERROR] Unable to create nvdsosd\n")

    # --- Sink: Video Renderer ---
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write("[ERROR] Unable to create Sink\n")
    sink.set_property('sync', False)

    # Add all elements into the pipeline
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link static elements
    streammux.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Dynamic pad linking (source pad of decodebin to sink pad of streammux)
    def decodebin_child_added(child_proxy, Object, name, user_data):
        print(f"[INFO] Decodebin child added: {name}")
        if name.find("decodebin") != -1:
            Object.connect("child-added", decodebin_child_added, user_data)

    def cb_newpad(decodebin, decoder_src_pad, data):
        print("[INFO] In cb_newpad")
        caps = decoder_src_pad.query_caps(None)
        gstname = caps.to_string()
        print(f"[INFO] gstname = {gstname}")

        if gstname.find("video") != -1:
            sinkpad = streammux.get_request_pad("sink_0")
            if not sinkpad:
                sys.stderr.write("[ERROR] Unable to get the sink pad of streammux\n")
            decoder_src_pad.link(sinkpad)
        else:
            print("[INFO] It has type: ", gstname)

    source.connect("pad-added", cb_newpad, streammux)
    source.connect("child-added", decodebin_child_added, streammux)

    pipeline.add(source)

    # Create event loop and feed GStreamer bus messages
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start playing
    print("[INFO] Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception:
        pass

    # Cleanup
    print("[INFO] Exiting app")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <URI or filepath>\n")
        sys.exit(1)

    sys.exit(main(sys.argv))
