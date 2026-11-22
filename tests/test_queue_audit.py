from lsc70 import queue_audit as qa


def test_a_picture_is_not_counted_as_an_isolated_frame():
    lm = [[0.1 * j, 0.1 * j, 0.0] for j in range(21)]
    video = {"video_id": "v", "frame": 0, "t_s": 0.0, "landmarks": lm}
    picture = {**video, "video_id": "img", "media": "image"}
    assert qa.junk_flags([video], {"v:0": 0.0}) == {"v:0": {"isolated"}}
    assert qa.junk_flags([picture], {"img:0": 0.0}) == {"img:0": set()}
