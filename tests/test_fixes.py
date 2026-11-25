"""六类缺陷的行为回归测试。

``test_spotlight_shadow_atlas.py`` 只验证公开入口“不是 None”。这里针对题目
描述的每个故障模式做可证伪的断言，防止实现退化回“能跑但错”的状态：

1. 页分配按灯 ID 稳定映射（与可见性 / 提交顺序无关，跨帧不换页）；
2. 销毁即归还页，并把相邻空闲页合并；
3. 写入前等待异步分配完成；
4. 自适应分辨率按页组统一（相邻页边界分辨率一致）；
5. 采样考虑图集翻转（页内偏移不被二次反转）；
6. 页边界有 guard band，采样被夹在内容区内。
"""

import random

import pytest

import graphics.spotlight_shadow_atlas as mod


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------

def make_atlas(page_count=16, atlas_size=4096, guard=2):
    return mod.SpotlightShadowAtlas(
        page_count=page_count, atlas_size=atlas_size, guard=guard
    )


def mapping(result):
    return {int(light): int(page) for light, page in result["mapping"].items()}


# ---------------------------------------------------------------------------
# 1. 页分配按灯 ID 稳定映射
# ---------------------------------------------------------------------------

def test_stable_page_depends_only_on_light_id():
    assert int(mod.stable_page(42)) == int(mod.stable_page(42))
    # 与页数无关的“同一盏灯 → 同一页”只要求确定，不要求对页数不变。
    assert int(mod.stable_page(42, 16)) == int(mod.stable_page(42, 16))


def test_stable_page_is_a_bijection_for_integer_ids():
    pages = [int(mod.stable_page(light)) for light in range(mod.DEFAULT_PAGE_COUNT)]
    assert len(set(pages)) == mod.DEFAULT_PAGE_COUNT


def test_stable_page_negative_id_is_in_range():
    for light in (-1, -2, -100):
        assert 0 <= int(mod.stable_page(light)) < mod.DEFAULT_PAGE_COUNT


def test_stable_page_ignores_input_order():
    lights = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    forward = {int(r["light_id"]): int(r) for r in mod.stable_page(lights)}
    shuffled = [lights[2], lights[0], lights[3], lights[1]]
    backward = {int(r["light_id"]): int(r) for r in mod.stable_page(shuffled)}
    assert forward == backward


def test_page_does_not_move_when_other_lights_come_and_go():
    atlas = make_atlas()
    first = mapping(mod.atlas_alloc([{"id": 7}], atlas=atlas))
    # 新灯加入、旧灯消失，7 号灯必须留在原页。
    mod.atlas_alloc([{"id": 9}, {"id": 10}, {"id": 11}], atlas=atlas)
    again = mapping(mod.atlas_alloc([{"id": 7}], atlas=atlas))
    assert again[7] == first[7]


def test_page_is_stable_across_random_visibility_over_frames():
    rng = random.Random(20240501)
    ids = list(range(16))
    atlas = make_atlas()
    seen = {}
    for _frame in range(8):
        visible = [
            {"id": i, "visible": rng.random() < 0.6}
            for i in ids
            if rng.random() < 0.85
        ]
        rng.shuffle(visible)
        for light, page in mapping(mod.atlas_alloc(visible, atlas=atlas)).items():
            assert seen.setdefault(light, page) == page


def test_page_stable_flag_reports_real_stickiness():
    atlas = make_atlas()
    result = mod.atlas_alloc([{"id": 1}, {"id": 2}], atlas=atlas)
    assert result["page_stable"] is True


# ---------------------------------------------------------------------------
# 2. 销毁即归还页
# ---------------------------------------------------------------------------

def test_destroy_returns_page_immediately():
    atlas = make_atlas(page_count=8)
    mapping(mod.atlas_alloc([{"id": 5}], atlas=atlas))
    page = atlas.page_of(5)
    release = atlas.release(5)
    assert release["released"] is True
    assert release["immediate"] is True
    assert page in atlas.free_pages
    assert page not in atlas.light_pages.values()


def test_destroy_coalesces_adjacent_free_pages():
    # 先占满 0..7，再依次销毁，空闲区间应合并成一整段。
    atlas = make_atlas(page_count=8)
    mapping(mod.atlas_alloc([{"id": i} for i in range(8)], atlas=atlas))
    for light in range(8):
        atlas.release(light)
    assert atlas.free_ranges == [(0, 7)]
    assert atlas.fragmentation == 0.0


def test_release_page_stateless_merges_ranges():
    result = mod.release_page(1, free=[5, 6, 7])
    page = int(mod.stable_page(1))
    if page == 4:
        assert result["coalesced"] is True
    assert result["free_ranges"]
    assert result["immediate"] is True


def test_freed_page_is_reused():
    atlas = make_atlas(page_count=2)
    mapping(mod.atlas_alloc([{"id": 0}, {"id": 1}], atlas=atlas))
    atlas.release(0)
    assert len(atlas.free_pages) == 1
    # 再申请一盏新灯必须能拿到被归还的页，而不是报池耗尽。
    result = mod.atlas_alloc([{"id": 99}], atlas=atlas)
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# 3. 写入前等待分配完成
# ---------------------------------------------------------------------------

def test_write_before_alloc_is_rejected():
    atlas = make_atlas(page_count=4, atlas_size=2048)
    result = mod.atlas_alloc([{"id": 3}], atlas=atlas, async_alloc=True)
    page = mapping(result)[3]
    assert atlas.pages[page]["alloc_state"] == mod.PENDING
    with pytest.raises(mod.ShadowAtlasError):
        atlas.write(page)
    assert atlas.metrics["write_before_alloc"] == 1


def test_wait_alloc_allows_write():
    atlas = make_atlas(page_count=4, atlas_size=2048)
    result = mod.atlas_alloc([{"id": 3}], atlas=atlas, async_alloc=True)
    page = mapping(result)[3]
    wait = mod.wait_alloc(page, atlas=atlas)
    assert wait["allocated"] is True
    assert wait["waited"] is True
    assert atlas.write(page)["ok"] is True


def test_wait_alloc_transitions_pending_to_allocated():
    atlas = make_atlas(page_count=4, atlas_size=2048)
    page = mapping(mod.atlas_alloc([{"id": 3}], atlas=atlas, async_alloc=True))[3]
    mod.wait_alloc(page, atlas=atlas)
    assert atlas.pages[page]["alloc_state"] == mod.ALLOCATED
    assert atlas.metrics["pending_allocs"] == 0


def test_stateless_wait_alloc_waits_out_pending():
    report = mod.wait_alloc({"page": 2, "alloc_state": mod.PENDING})
    assert report["allocated"] is True
    assert report["waited"] is True
    assert report["pending"] is False


def test_destroying_pending_light_clears_pending_count():
    atlas = make_atlas(page_count=4, atlas_size=2048)
    mod.atlas_alloc([{"id": 3}], atlas=atlas, async_alloc=True)
    assert atlas.metrics["pending_allocs"] == 1
    atlas.release(3)
    assert atlas.metrics["pending_allocs"] == 0


# ---------------------------------------------------------------------------
# 4. 页组分辨率统一
# ---------------------------------------------------------------------------

def test_page_group_res_takes_one_value_for_the_whole_group():
    report = mod.page_group_res([128, 256, 200])
    assert report["uniform"] is True
    assert len(set(report["per_page"])) == 1
    assert report["resolution"] == 256  # 取最大需求 256，已落在 2 的幂阶梯上


def test_page_group_res_quantizes_to_power_of_two():
    value = mod.page_group_res([300])["resolution"]
    assert value & (value - 1) == 0
    assert value >= 256


def test_adjacent_pages_in_a_group_share_resolution():
    atlas = make_atlas()
    mod.atlas_alloc(
        [
            {"id": 0, "resolution": 64},
            {"id": 1, "resolution": 512},
            {"id": 2, "resolution": 1024},
        ],
        atlas=atlas,
    )
    by_group = {}
    for page, record in atlas.pages.items():
        if record["alloc_state"] != mod.FREE:
            by_group.setdefault(record["group"], set()).add(record["resolution"])
    assert all(len(values) == 1 for values in by_group.values())


def test_group_resolution_stays_uniform_when_pages_arrive_late():
    atlas = make_atlas()
    mod.atlas_alloc([{"id": 0, "resolution": 64}], atlas=atlas)
    mod.atlas_alloc([{"id": 3, "resolution": 1024}], atlas=atlas)
    by_group = {}
    for page, record in atlas.pages.items():
        if record["alloc_state"] != mod.FREE:
            by_group.setdefault(record["group"], set()).add(record["resolution"])
    assert all(len(values) == 1 for values in by_group.values())


# ---------------------------------------------------------------------------
# 5. 采样考虑图集翻转
# ---------------------------------------------------------------------------

def test_flip_sample_maps_page_top_to_high_v():
    rect = (0, 0, 1024, 1024)
    size = 4096
    top = mod.flip_sample(rect, (0.5, 0.0), flip_y=True, atlas_size=size)
    bottom = mod.flip_sample(rect, (0.5, 1.0), flip_y=True, atlas_size=size)
    # 页顶在翻转后的图集里应该对应更高的 V。
    assert top["v"] > bottom["v"]
    assert top["v"] == pytest.approx(1.0)
    assert bottom["v"] == pytest.approx(1.0 - 1024 / size)


def test_flip_sample_equals_unflipped_with_v_mirrored():
    rect = (1024, 0, 1024, 1024)
    for uv in ((0.0, 0.0), (0.25, 0.75), (1.0, 1.0), (0.5, 0.5)):
        flipped = mod.flip_sample(rect, uv, flip_y=True, atlas_size=4096)
        plain = mod.flip_sample(rect, uv, flip_y=False, atlas_size=4096)
        assert flipped["u"] == pytest.approx(plain["u"])
        assert flipped["v"] == pytest.approx(1.0 - plain["v"])


def test_flip_is_applied_once_not_twice():
    # 只翻转一次时，翻转后的 V 必须等于未翻转 V 的镜像；若在页内又翻一次
    # （或根本没翻），这个恒等式不成立。
    rect = (0, 0, 1024, 1024)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        plain = mod.flip_sample(rect, (0.5, v), flip_y=False, atlas_size=4096)
        flipped = mod.flip_sample(rect, (0.5, v), flip_y=True, atlas_size=4096)
        assert flipped["v"] == pytest.approx(1.0 - plain["v"])
    # 页顶（v=0）在翻转后应是图集最上方（V=1），页底（v=1）是页底。
    top = mod.flip_sample(rect, (0.5, 0.0), flip_y=True, atlas_size=4096)
    assert top["v"] == pytest.approx(1.0)


def test_sample_uses_flip_by_default():
    report = mod.sample(0, (0.5, 0.5), guard=2, atlas_size=4096, page_count=16)
    assert report["flipped"] is True


# ---------------------------------------------------------------------------
# 6. 页边界 guard band
# ---------------------------------------------------------------------------

def test_guard_band_insets_content_rect():
    report = mod.guard_band(0, 8, page_rect=(0, 0, 1024, 1024), atlas_size=4096)
    assert report["guard"] == 8
    assert report["content_rect"] == (8.0, 8.0, 1008.0, 1008.0)
    assert report["valid"] is True


def test_guard_band_clamps_oversized_request():
    report = mod.guard_band(0, 10_000, page_rect=(0, 0, 1024, 1024), atlas_size=4096)
    # 不允许 guard 吃掉半个页，内容区必须仍然存在。
    assert report["guard"] * 2 < 1024
    assert report["valid"] is True
    assert report["clamped"] is True


def test_sampling_is_clamped_inside_guard_band():
    guard = 16
    for uv in ((-5, -5), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)):
        report = mod.sample(0, uv, guard=guard, atlas_size=4096, page_count=16)
        inset_u, inset_v = report["page_uv"]
        band = guard / 1024.0
        assert band - 1e-12 <= inset_u <= 1.0 - band + 1e-12
        assert band - 1e-12 <= inset_v <= 1.0 - band + 1e-12


def test_guard_band_zero_keeps_full_page():
    report = mod.guard_band(0, 0, page_rect=(0, 0, 512, 512), atlas_size=4096)
    assert report["guard"] == 0
    assert report["content_rect"] == (0.0, 0.0, 512.0, 512.0)


# ---------------------------------------------------------------------------
# 组合：一帧渲染
# ---------------------------------------------------------------------------

def test_render_frame_is_consistent_and_consumes_pending():
    report = mod.shadow_render(1)
    assert report["ok"] is True
    assert report["pending_pages"] == []
    assert report["group_uniform"] is True
    assert report["write_errors"] == []
    assert all(sample["flipped"] for sample in report["samples"])


def test_determinism_across_order_and_visibility():
    report = mod.determinism(1)
    assert report["deterministic"] is True
    assert report["ok"] is True


def test_validation_accepts_a_clean_atlas():
    atlas = make_atlas()
    mod.atlas_alloc([{"id": i} for i in range(4)], atlas=atlas)
    report = mod.validation(atlas)
    assert report["ok"] is True
    assert report["errors"] == []


def test_audit_and_metrics_are_reproducible():
    atlas = make_atlas()
    mod.atlas_alloc([{"id": i} for i in range(4)], atlas=atlas)
    assert mod.audit(atlas)["digest"] == mod.audit(atlas)["digest"]
    snapshot = mod.metrics(atlas)
    assert snapshot["pages"] == atlas.page_count
    assert snapshot["violation_free"] is True


def test_validation_flags_group_resolution_regression():
    atlas = make_atlas()
    mod.atlas_alloc([{"id": i} for i in range(4)], atlas=atlas)
    live = [
        page for page, record in atlas.pages.items() if record["alloc_state"] != mod.FREE
    ]
    group = atlas.pages[live[0]]["group"]
    same_group = [p for p in live if atlas.pages[p]["group"] == group]
    if len(same_group) >= 2:
        atlas.pages[same_group[0]]["resolution"] = 64
        atlas.pages[same_group[1]]["resolution"] = 128
        codes = {issue["code"] for issue in mod.validation(atlas)["errors"]}
        assert "group_resolution_mismatch" in codes


def test_validation_flags_write_before_alloc():
    atlas = make_atlas()
    mod.atlas_alloc([{"id": 0}], atlas=atlas)
    atlas.metrics["write_before_alloc"] = 1
    codes = {issue["code"] for issue in mod.validation(atlas)["errors"]}
    assert "write_before_alloc" in codes


def test_validation_flags_guard_clamp_as_single_warning():
    atlas = mod.SpotlightShadowAtlas(page_count=64, atlas_size=1024, guard=40)
    mod.atlas_alloc([{"id": 0}], atlas=atlas)
    issues = mod.validation(atlas)["issues"]
    clamped = [issue for issue in issues if issue["code"] == "guard_band_clamped"]
    assert len(clamped) == 1
    assert clamped[0]["severity"] == "warning"

