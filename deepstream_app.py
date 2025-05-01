import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds  # DeepStream Python bindings

# Initialize GStreamer
Gst.init(None)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        print(f"Processing frame {frame_meta.frame_num}")
        print(f"Source Frame resolution: {frame_meta.source_frame_width}x{frame_meta.source_frame_height}")
        l_obj = frame_meta.obj_meta_list
        obj_count = 0

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                obj_count += 1

                # Verify detection matches parser output
                print(f"  Object {obj_count}:")
                print(f"    Class: {obj_meta.class_id}")
                print(f"    Confidence: {obj_meta.confidence}")
                print(f"    BBox: ({obj_meta.rect_params.left}, {obj_meta.rect_params.top}, "
                      f"{obj_meta.rect_params.width}, {obj_meta.rect_params.height})")

                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green
                obj_meta.text_params.display_text = f"Face {obj_meta.confidence:.2f}"

            except StopIteration:
                break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print(f"Frame {frame_meta.frame_num} had {obj_count} faces")
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(source_uri):
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    source = Gst.ElementFactory.make("uridecodebin", "source")
    source.set_property("uri", source_uri)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 2560)
    streammux.set_property("height", 1440)
    # streammux.set_property("enable-padding", 1)
    streammux.set_property("batched-push-timeout", 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config_infer_primary_scrfd.txt")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not all([source, streammux, pgie, nvvidconv, nvosd, sink]):
        sys.stderr.write(" Unable to create elements \n")
        return

    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    source.connect("pad-added", cb_newpad, streammux)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    osdsinkpad = nvosd.get_static_pad("sink")
    if osdsinkpad:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("[INFO] Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()

    def bus_call(bus, message, user_data):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[INFO] End-of-stream reached")
            user_data.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            sys.stderr.write(f"[ERROR] {err.message}\n{dbg or ''}\n")
            user_data.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    try:
        loop.run()
    except:
        pass

    print("[INFO] Exiting app")
    pipeline.set_state(Gst.State.NULL)

def cb_newpad(decodebin, pad, data):
    print("[INFO] In cb_newpad")

    caps = pad.get_current_caps()
    gstname = caps.to_string()
    print("[INFO] gstname =", gstname)

    if "video" in gstname:
        sinkpad = data.get_request_pad("sink_0")
        if sinkpad:
            pad.link(sinkpad)
        else:
            sys.stderr.write(" Unable to get sink pad of streammux \n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s <URI>\n" % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])
