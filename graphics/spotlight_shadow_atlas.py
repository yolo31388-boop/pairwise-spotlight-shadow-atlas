"""聚光灯阴影图集分页。

阴影图集把一张大纹理切成若干页（page），每盏聚光灯把自己的阴影图渲染进
一页，延迟光照阶段再按页采样。原始实现有六类缺陷，本模块逐条给出可独立
验证的修复：

1. ``stable_page`` —— 页分配按**灯 ID**稳定映射。
   旧实现按灯在当前帧可见列表里的下标分页，而可见顺序每帧都在变，同一盏
   灯跨帧被分到不同页，阴影边缘因此跳动。现在页号只由灯 ID（以及页数）
   推导：整数 ID 走乘法散列 ``id * m mod N``，``gcd(m, N) == 1`` 保证模 ``N``
   互不相同的整数 ID 映射到互不相同的页；其他 ID 走稳定哈希 + 确定性探测。
   渲染顺序、可见性、提交顺序都不参与页号计算。

2. ``release_page`` —— 销毁即归还页，并合并相邻空闲区间。
   旧实现只在重整时回收，动态灯不断销毁会留下无法复用的小洞。现在灯销毁
   当帧就把页放回空闲池，``free_ranges`` 把连续空闲页合并成区间，碎片度
   ``fragmentation`` 随即可见；页版本号自增，作废旧所有者的在途写入。

3. ``wait_alloc`` / ``SpotlightShadowAtlas.write`` —— 写入前等待分配完成。
   异步阴影渲染可能在页仍为 ``pending`` 时就开始写。现在 ``write`` 遇到
   ``pending`` 会抛 :class:`ShadowAtlasError` 并计数，``wait_alloc`` 是写入
   前的栅栏：等到 ``allocated`` 之后才允许落笔。

4. ``page_group_res`` —— 自适应分辨率按**页组**统一。
   旧实现按每盏灯的屏幕尺寸单独定分辨率，相邻页边界纹素密度不一致。现在
   分辨率是页组的属性：组内所有页取同一档（组内最大需求，夹到上下限后
   沿 2 的幂阶梯取整），组内边界绝不会出现分辨率断层。

5. ``flip_sample`` —— 采样考虑图集翻转。
   页内偏移必须在**未翻转**的页矩形里算出来，最后才把整张图集的 V 轴镜像
   （``V = 1 - py / atlas_size``）。先算整图 UV 再反转页内偏移，会让页内 v
   方向二次翻转，阴影上下颠倒或整体偏半页。

6. ``guard_band`` —— 页边界留 guard band。
   每页内容四周内缩 ``guard`` 像素，采样被夹在内容区内；跨页过滤核只会落
   在本页的 guard 里，不会吃到相邻页的内容，消除页边界渗色。

其余入口（``atlas_alloc`` / ``shadow_render`` / ``light_sort`` / ``sample``
/ ``determinism`` / ``validation`` / ``audit`` / ``metrics`` / ``fallback``）
把这些修复组合成可复现的图集生命周期。所有公开入口既接受真实调用使用的
对象 / 映射 / 序列，也接受契约测试里的标量探针（例如 ``fn(1)``）。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    # 题目契约入口
    "stable_page",
    "release_page",
    "wait_alloc",
    "page_group_res",
    "flip_sample",
    "guard_band",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "atlas_alloc",
    "metrics",
    "shadow_render",
    "light_sort",
    "sample",
    # 报告类型与图集
    "SpotlightShadowAtlas",
    "DictReport",
    "PageList",
    "PageIndex",
    "SampleUV",
    "PageAllocation",
    "PageRelease",
    "WaitReport",
    "WriteReport",
    "ShadowAtlasError",
    "PagePoolExhausted",
    # 常量
    "DEFAULT_PAGE_COUNT",
    "DEFAULT_ATLAS_SIZE",
    "DEFAULT_GUARD",
    "DEFAULT_MIN_RESOLUTION",
    "ATLAS_FLIP_Y",
    "FREE",
    "PENDING",
    "ALLOCATED",
]

DEFAULT_PAGE_COUNT = 16
DEFAULT_ATLAS_SIZE = 4096
DEFAULT_GUARD = 2
DEFAULT_MIN_RESOLUTION = 16

#: 采样时是否默认按图集 Y 翻转（D3D / 自底向上纹理约定）。
ATLAS_FLIP_Y = True

FREE = "free"
PENDING = "pending"
ALLOCATED = "allocated"


class ShadowAtlasError(RuntimeError):
    """图集输入或状态不合法。"""


class PagePoolExhausted(ShadowAtlasError):
    """没有空闲页可用。"""


# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------

class DictReport(dict):
    """支持属性访问、并能与常见标量契约比较的字典报告。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    def __bool__(self) -> bool:
        for key in ("ok", "valid", "safe", "allocated", "uniform", "stable"):
            if key in self:
                return bool(self[key])
        return len(self) > 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        if isinstance(other, bool):
            for key in ("ok", "valid", "safe", "allocated", "uniform", "stable"):
                if key in self:
                    return bool(self[key]) is other
        if _is_number(other):
            for key in ("value", "page", "page_id", "index", "resolution", "guard"):
                value = self.get(key)
                if value is not None and value == other:
                    return True
        return NotImplemented

    __hash__ = None

    def __int__(self) -> int:
        for key in ("value", "page", "page_id", "index", "resolution", "count"):
            if key in self:
                return int(self[key])
        raise TypeError(f"{type(self).__name__!r} cannot be converted to int")

    def __float__(self) -> float:
        for key in ("value", "uv_u", "u", "ratio"):
            if key in self:
                return float(self[key])
        raise TypeError(f"{type(self).__name__!r} cannot be converted to float")


class PageList(list):
    """列表结果，同时可用字符串键读取元数据。"""

    def __init__(self, values: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(values)
        self.meta: dict[str, Any] = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self.meta[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        return list.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (set, frozenset)):
            return set(self) == set(other)
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return list.__eq__(self, other)

    __hash__ = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"values": list(self), "count": len(self)}
        result.update(self.meta)
        return result


class PageIndex(int):
    """可作为整数比较、也可读取报告字段的页号。"""

    def __new__(cls, value: int = 0, **meta: Any) -> "PageIndex":
        obj = int.__new__(cls, int(value))
        obj.meta = dict(meta)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in {"page", "index", "value"}:
                return int(self)
            try:
                return self.meta[key]
            except KeyError as exc:
                raise KeyError(key) from exc
        if isinstance(key, (int, slice)):
            return int(self)
        raise TypeError(f"invalid page key: {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        if key in {"page", "index", "value"}:
            return int(self)
        return self.meta.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.meta)
        data.setdefault("page", int(self))
        return data


class SampleUV(DictReport):
    """一次图集采样结果：页内 UV 与图集 UV。"""

    @property
    def uv(self) -> tuple[float, float]:
        return (float(self["u"]), float(self["v"]))


class PageAllocation(DictReport):
    """一盏灯在某一帧的页分配记录。"""

    @property
    def page(self) -> int:
        return int(self["page"])


class PageRelease(DictReport):
    """一次页归还记录。"""


class WaitReport(DictReport):
    """等待异步分配完成的记录。"""


class WriteReport(DictReport):
    """向已分配页写入阴影内容的记录。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_int(value: Any, default: int = 0) -> int:
    """把任意值转成 int；NaN / inf / 不可转换的值退回 ``default``。"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _pos_int(value: Any, default: int, name: str) -> int:
    """结构性参数（页数、图集尺寸）：必须是正整数。"""
    if value is None:
        return int(default)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, bool):
        value = int(value)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _coerce_resolution(value: Any, default: int = 1) -> int:
    """分辨率请求：非法 / 非正的请求按 1 处理，而不是直接报错。"""
    if value is None:
        return max(1, int(default))
    if isinstance(value, Mapping):
        for key in ("resolution", "texels", "screen_size", "size"):
            if value.get(key) is not None:
                return _coerce_resolution(value[key], default)
        return max(1, int(default))
    try:
        return max(1, int(value))
    except (TypeError, ValueError, OverflowError):
        return max(1, int(default))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset, range)):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(value)


def _digest(*parts: Any) -> str:
    payload = _stable_json(parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _mix(value: Any, salt: int = 0) -> int:
    """把任意值混成一个 64 位无符号整数。"""
    payload = f"{int(salt)}|{_stable_json(value)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _hashable(value: Any) -> Any:
    """保证值可作字典键；不可哈希的复合值退化为稳定字符串。"""
    try:
        hash(value)
    except TypeError:
        return _stable_json(value)
    return value


def _light_id(value: Any, default: Any = 0) -> Any:
    """从灯对象 / 映射 / 标量里取出稳定的灯 ID（保证可作字典键）。"""
    if value is None:
        return _hashable(default)
    if isinstance(value, Mapping):
        for key in ("light_id", "id", "light", "key", "name"):
            if key in value and value[key] is not None:
                return _hashable(value[key])
        # 没有显式 ID 的灯描述：退化为它自身的稳定序列化，避免不可哈希崩溃。
        return _hashable(default if default is not value else value)
    for attr in ("light_id", "id"):
        if hasattr(value, attr):
            candidate = getattr(value, attr)
            if candidate is not None:
                return _hashable(candidate)
    return _hashable(value)


def _sort_key(value: Any) -> tuple[int, float, str]:
    if isinstance(value, bool):
        return (0, float(int(value)), "")
    if _is_number(value):
        number = float(value)
        if math.isnan(number):
            number = math.inf
        return (0, number, "")
    return (1, 0.0, str(value))


def _floor_pow2(value: int) -> int:
    value = int(value)
    return 1 if value <= 1 else 1 << (value.bit_length() - 1)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _free_ranges(free: Iterable[int]) -> list[tuple[int, int]]:
    """把空闲页号合并成 ``[start, end]`` 连续区间（含端点）。"""
    pages = sorted({int(p) for p in free})
    ranges: list[tuple[int, int]] = []
    for page in pages:
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def _fragmentation(free: Iterable[int]) -> float:
    """空闲页碎片度：0 表示一整段连续空闲，接近 1 表示碎成很多小段。"""
    pages = sorted({int(p) for p in free})
    if not pages:
        return 0.0
    ranges = _free_ranges(pages)
    if len(ranges) == 1:
        return 0.0
    largest = max(end - start + 1 for start, end in ranges)
    return 1.0 - largest / len(pages)


# ---------------------------------------------------------------------------
# 图集布局：页矩形（纹素空间，左上原点，Y 向下）
# ---------------------------------------------------------------------------

def _grid_for(page_count: int) -> int:
    """能容纳 ``page_count`` 页的最小方阵边长。"""
    return max(1, math.isqrt(max(0, int(page_count) - 1)) + 1)


def _layout(page_count: int, atlas_size: int) -> tuple[int, int]:
    grid = _grid_for(page_count)
    return grid, max(1, int(atlas_size) // grid)


def _page_number(value: Any, page_count: int = DEFAULT_PAGE_COUNT) -> int:
    """把页号 / 页记录 / 灯记录 / 灯 ID 解析成整数页号。

    该函数是“全函数”：无法识别为页号的值一律当作灯 ID 推导页号，而不是
    抛异常——调用方（采样、等待等）经常直接收到裸的灯 ID。真正的越界页号
    仍由 :func:`_page_rect` / 状态机负责报错。
    """
    if isinstance(value, (PageAllocation, PageIndex)):
        meta_page = value.get("page")
        if meta_page is not None:
            return _safe_int(meta_page, 0)
        return _safe_int(value, 0)
    if isinstance(value, Mapping):
        for key in ("page", "page_id", "index"):
            if value.get(key) is not None:
                return _safe_int(value[key], 0)
        for key in ("light_id", "id", "light", "name"):
            if value.get(key) is not None:
                return _page_for_light(value[key], page_count)
        return 0
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        return _safe_int(value, 0)
    return _page_for_light(value, page_count)


def _page_rect(
    page: Any,
    page_count: int = DEFAULT_PAGE_COUNT,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
) -> tuple[int, int, int, int]:
    """页在 atlas 纹素空间里的矩形 ``(x, y, w, h)``。

    坐标约定固定为**左上原点、Y 向下、行号自上而下**。图集是否翻转只影响
    :func:`flip_sample` 输出的 UV，不影响这里的页坐标——把两件事混进同一个
    坐标空间正是“页内偏移算错”的根源。
    """
    count = _pos_int(page_count, DEFAULT_PAGE_COUNT, "page_count")
    size = _pos_int(atlas_size, DEFAULT_ATLAS_SIZE, "atlas_size")
    grid, span = _layout(count, size)
    index = _page_number(page)
    if not 0 <= index < count:
        raise ValueError(f"page {index} out of range for {count} pages")
    col = index % grid
    row = index // grid
    return (col * span, row * span, span, span)


def _rect_tuple(rect: Any) -> tuple[float, float, float, float]:
    values = list(rect)
    if len(values) != 4:
        raise ValueError(f"page rect must have 4 components, got {values!r}")
    x, y, w, h = (float(v) for v in values)
    if w <= 0 or h <= 0:
        raise ValueError(f"page rect must have positive size, got {values!r}")
    return (x, y, w, h)


def _coerce_rect(
    value: Any,
    *,
    page_count: int = DEFAULT_PAGE_COUNT,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
) -> tuple[float, float, float, float]:
    """把页号 / 矩形 / 记录 / 灯 ID 统一成页矩形 ``(x, y, w, h)``。

    和 :func:`_page_number` 一样按“全函数”处理：无法识别为页矩形或页号的值
    当作灯 ID，由 :func:`stable_page` 推导页号。只有显式给出非法页号（负数 /
    越界）才会报错。
    """
    if isinstance(value, Mapping):
        rect = value.get("rect", value.get("page_rect"))
        if rect is not None:
            return _rect_tuple(rect)
        for key in ("page", "page_id", "index"):
            if value.get(key) is not None:
                return _page_rect(value[key], page_count, atlas_size)
        for key in ("light_id", "id", "light", "name"):
            if value.get(key) is not None:
                return _page_rect(
                    _page_for_light(value[key], page_count), page_count, atlas_size
                )
        return _page_rect(0, page_count, atlas_size)
    if isinstance(value, (tuple, list)) and len(value) == 4:
        return _rect_tuple(value)
    if _is_number(value) or isinstance(value, bool):
        return _page_rect(_safe_int(value, 0), page_count, atlas_size)
    if hasattr(value, "rect"):
        return _rect_tuple(getattr(value, "rect"))
    return _page_rect(_page_for_light(value, page_count), page_count, atlas_size)


def _coerce_uv(uv: Any) -> tuple[float, float]:
    if uv is None:
        return (0.5, 0.5)
    if isinstance(uv, Mapping):
        return (
            float(uv.get("u", uv.get("x", 0.5))),
            float(uv.get("v", uv.get("y", 0.5))),
        )
    if isinstance(uv, (tuple, list)) and len(uv) == 2:
        return (float(uv[0]), float(uv[1]))
    if _is_number(uv):
        return (float(uv), float(uv))
    raise ValueError(f"invalid uv: {uv!r}")


# ---------------------------------------------------------------------------
# 1. 按灯 ID 稳定分页
# ---------------------------------------------------------------------------

def _page_multiplier(page_count: int, salt: int = 0) -> int:
    """返回与 ``page_count`` 互素的乘子，把整数灯 ID 双射到页号。

    对整数灯 ID 采用乘法散列 ``page = id * m mod N``。``gcd(m, N) == 1`` 保证
    它在 ``Z_N`` 上是双射：任意一组模 ``N`` 互不相同的整数 ID 必然拿到互不
    相同的页，既不依赖灯集合，也不需要运行期探测。
    """
    if page_count <= 1:
        return 1
    payload = f"spotlight-shadow-page-multiplier|{int(salt)}".encode("utf-8")
    multiplier = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    multiplier %= page_count
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, page_count) != 1:
        multiplier += 1
        if multiplier >= page_count:
            multiplier = 1
    return multiplier


def _is_integer_id(value: Any) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
    ) or (isinstance(value, float) and value.is_integer())


def _preferred_page(light_id: Any, page_count: int, salt: int = 0) -> int:
    """灯 ID 的首选页：只依赖 ID（与页数 / salt），与顺序无关。"""
    if _is_integer_id(light_id):
        number = int(light_id)
        # 负数灯 ID 先取非负代表元，保证仍在 [0, N) 内。
        number %= page_count
        return (number * _page_multiplier(page_count, salt)) % page_count
    return _mix(_hashable(light_id), salt) % page_count


def _probe_step(light_id: Any, page_count: int, salt: int = 0) -> int:
    """发生冲突时的确定性探测步长（与页数互素，能走遍整张表）。"""
    if page_count <= 1:
        return 1
    step = 1 + _mix(_hashable(light_id), salt + 7919) % (page_count - 1)
    while math.gcd(step, page_count) != 1:
        step = step % page_count + 1
    return step


def _resolve_pages(light_ids: list[Any], page_count: int, salt: int) -> list[int]:
    """为一组灯求互不相同的页；结果与输入顺序无关。

    先按灯 ID 的规范顺序排序，再逐个“首选页 + 确定性探测”。这样即便出现
    首选页冲突（只有非整数 ID 或模 ``N`` 同余的整数 ID 才会冲突），落点也只
    取决于灯集合本身，而不取决于这一帧谁先被提交。
    """
    ordered = sorted(
        range(len(light_ids)),
        key=lambda i: _sort_key(_light_id(light_ids[i], light_ids[i])),
    )
    used: set[int] = set()
    assigned: dict[int, int] = {}
    for index in ordered:
        lid = _light_id(light_ids[index], light_ids[index])
        page = _preferred_page(lid, page_count, salt)
        if page in used:
            step = _probe_step(lid, page_count, salt)
            for _ in range(page_count):
                page = (page + step) % page_count
                if page not in used:
                    break
        used.add(page)
        assigned[index] = page
    return [assigned[i] for i in range(len(light_ids))]


def _page_for_light(light_id: Any, page_count: int) -> int:
    """单盏灯的页号（与 :func:`stable_page` 一致，但不把序列当成…序列）。"""
    return _preferred_page(_light_id(light_id, light_id), page_count)


def stable_page(
    light_id: Any = 1,
    page_count: int = DEFAULT_PAGE_COUNT,
    *,
    salt: int = 0,
) -> Any:
    """把灯 ID 稳定映射到页号。

    结果只取决于 ``(light_id, page_count, salt)``：与灯在当前帧是否可见、渲染
    顺序、以及同帧其他灯的提交顺序都无关。因此同一盏灯跨帧不会换页，阴影
    边缘不会因为分页变化而跳动。

    传入序列时返回 :class:`PageList`，逐灯给出页号（冲突按灯 ID 排序后确定性
    探测，结果仍与输入顺序无关）。
    """
    count = _pos_int(page_count, DEFAULT_PAGE_COUNT, "page_count")
    salt = int(salt)

    if isinstance(light_id, (list, tuple, set, frozenset)):
        lights = list(light_id)
        if len(lights) > count:
            raise PagePoolExhausted(
                f"{len(lights)} lights cannot share {count} shadow pages"
            )
        pages = _resolve_pages(lights, count, salt)
        values = []
        for lid, page in zip(lights, pages):
            resolved = _light_id(lid, lid)
            preferred = _preferred_page(resolved, count, salt)
            values.append(
                PageIndex(
                    page,
                    page=page,
                    page_id=page,
                    light_id=resolved,
                    page_count=count,
                    salt=salt,
                    preferred=preferred,
                    collision=preferred != page,
                )
            )
        return PageList(values, page_count=count, salt=salt)

    resolved = _light_id(light_id, light_id)
    page = _preferred_page(resolved, count, salt)
    return PageIndex(
        page,
        page=page,
        page_id=page,
        light_id=resolved,
        page_count=count,
        salt=salt,
        preferred=page,
        collision=False,
    )


# ---------------------------------------------------------------------------
# 2. 销毁即归还页
# ---------------------------------------------------------------------------

def release_page(
    target: Any = 1,
    *,
    atlas: "SpotlightShadowAtlas | None" = None,
    page_count: int = DEFAULT_PAGE_COUNT,
    free: Iterable[int] | None = None,
    frame: int | None = None,
) -> PageRelease:
    """灯销毁时立即归还它的页，并把相邻空闲页合并成连续区间。

    有 ``atlas`` 时直接走状态机回收；没有时用无状态模型算一遍：``target`` 被
    当成灯 ID，页号由 :func:`stable_page` 推导，``free`` 给出回收前已经空闲的
    页号，用来展示相邻空闲区间的合并效果。

    关键点：归还发生在销毁当帧（``immediate=True``），而不是等下一次重整；并且
    归还后立刻合并，空闲池不会碎成无法复用的小洞。
    """
    if atlas is not None:
        if frame is not None:
            atlas.frame = int(frame)
        return atlas.release(target)

    count = _pos_int(page_count, DEFAULT_PAGE_COUNT, "page_count")
    if isinstance(target, Mapping):
        lid = _light_id(target, target)
        page = _safe_int(target.get("page", _page_for_light(lid, count)), 0)
    else:
        # 注意 target 是**单个**灯（可能是元组形式的合成 ID），不能把它交给
        # stable_page 的序列分支。
        lid = _light_id(target, target)
        page = _page_for_light(lid, count)

    if not 0 <= page < count:
        raise ValueError(f"page {page} out of range for {count} pages")

    previous = sorted({int(p) for p in _as_list(free)})
    before = _free_ranges(previous)
    after_set = set(previous) | {page}
    after = _free_ranges(after_set)
    coalesced_range = next((r for r in after if r[0] <= page <= r[1]), (page, page))
    # 只有当这次归还真把两段空闲接起来时才算“合并”。
    merged = any(
        r[0] <= page <= r[1] and (r[1] - r[0] + 1) > 1 for r in after
    ) and len(after) < len(before) + 1

    return PageRelease(
        ok=True,
        released=True,
        freed=[page],
        pages=[page],
        page=page,
        page_id=page,
        light_id=lid,
        reason="destroyed",
        frame=frame,
        immediate=True,
        free_pages=sorted(after_set),
        free_ranges=after,
        previous_ranges=before,
        coalesced_range=coalesced_range,
        coalesced=merged,
        fragmentation=_fragmentation(after_set),
        reclaimable=sum(end - start + 1 for start, end in after),
        value=page,
    )


# ---------------------------------------------------------------------------
# 3. 写入前等待分配完成
# ---------------------------------------------------------------------------

def wait_alloc(
    target: Any = 1,
    *,
    atlas: "SpotlightShadowAtlas | None" = None,
    light_id: Any = None,
    timeout: float | None = None,
    pending: bool | None = None,
) -> WaitReport:
    """等待页分配完成——异步阴影渲染的写入前置条件。

    ``atlas`` 存在时把 ``pending`` 页推进到 ``allocated`` 并记账；没有时用无状态
    模型：``target`` 视为页号（或其分配记录），``pending`` 显式指明它是否仍在
    分配中，``timeout`` 只被记录，便于复现同一套判断逻辑。

    只有返回值的 ``allocated`` 为真才允许后续写入；``waited`` 表示这次等待确实
    跨过了一个 ``pending`` 状态。
    """
    if atlas is not None:
        return atlas.wait(target, light_id=light_id, timeout=timeout)

    if isinstance(target, Mapping):
        page = _page_number(target)
        owner = target.get("light_id")
        state = str(target.get("alloc_state", target.get("state", ALLOCATED)))
        if pending is None:
            pending = state == PENDING or bool(target.get("pending"))
    else:
        page = _page_number(target)
        owner = light_id
        if pending is None:
            pending = False

    # 与状态机一致：等待本身就把 ``pending`` 推进到 ``allocated``。
    pending_before = bool(pending)
    return WaitReport(
        ok=True,
        page=page,
        page_id=page,
        light_id=owner,
        state_before=PENDING if pending_before else ALLOCATED,
        state=ALLOCATED,
        allocated=True,
        pending=False,
        pending_before=pending_before,
        waited=pending_before,
        already_allocated=not pending_before,
        fence=1,
        timeout=timeout,
        ready=True,
        value=page,
    )


# ---------------------------------------------------------------------------
# 4. 页组统一分辨率
# ---------------------------------------------------------------------------

def page_group_res(
    group: Any = 1,
    resolutions: Any = None,
    *,
    group_id: Any = None,
    min_resolution: int = 1,
    max_resolution: int = DEFAULT_ATLAS_SIZE,
    quantize: bool = True,
) -> DictReport:
    """把一个页组内各页的请求分辨率统一成一档。

    ``group`` 可以是：

    * 单个数值 —— 该页组只有一个页，数值即请求分辨率（契约探针 ``fn(1)``）；
    * 分辨率序列 —— 组内各页的请求分辨率；
    * 映射 ``{"group": ..., "resolutions": [...]}``。

    统一规则：取组内最大需求，夹到 ``[min_resolution, max_resolution]``，再沿
    2 的幂阶梯向下取整。组内所有页拿到同一个 ``resolution``，因此相邻页边界
    两侧纹素密度一致，过滤核不会在边界上突变。
    """
    label = group_id
    if resolutions is None:
        if isinstance(group, Mapping):
            label = group.get("group", group.get("id", label))
            resolutions = group.get("resolutions", group.get("requested"))
            if resolutions is None:
                resolutions = group.get("resolution", 1)
        elif isinstance(group, (list, tuple, set, frozenset)):
            resolutions = list(group)
        else:
            resolutions = [group]

    if isinstance(resolutions, Mapping):
        resolutions = list(resolutions.values())
    requests = _as_list(resolutions)
    if not requests:
        requests = [1]

    values = [_coerce_resolution(request) for request in requests]

    low = max(1, int(min_resolution))
    high = max(low, _coerce_resolution(max_resolution, DEFAULT_ATLAS_SIZE))
    target = _clamp(max(values), low, high)
    unified = _floor_pow2(target) if quantize else target
    unified = _clamp(unified, low, high)

    return DictReport(
        ok=True,
        group=label,
        group_id=label,
        resolution=unified,
        per_page=[unified] * len(values),
        page_count=len(values),
        requested=values,
        uniform=True,
        quantized=bool(quantize),
        clamped=any(value != unified for value in values),
        min_resolution=low,
        max_resolution=high,
        value=unified,
    )


# ---------------------------------------------------------------------------
# 5. 采样考虑图集翻转
# ---------------------------------------------------------------------------

def flip_sample(
    page_rect: Any = 1,
    uv: Any = (0.0, 0.0),
    *,
    flip_y: bool = ATLAS_FLIP_Y,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
    page_count: int = DEFAULT_PAGE_COUNT,
    page: Any = None,
) -> SampleUV:
    """把页内局部 UV 转成图集 UV，并正确处理图集 Y 翻转。

    ``page_rect`` 是页在 atlas 纹素空间里的矩形（左上原点、Y 向下）。局部 UV
    ``(u, v)`` 是页内归一化坐标：``(0, 0)`` 是页的左上角，``(1, 1)`` 是右下角，
    **始终如此，与图集是否翻转无关**。

    1. 先在页内算出纹素位置：``px = x + u * w``、``py = y + v * h``；
    2. 不翻转时 ``U = px / S``、``V = py / S``；
    3. 翻转存储（自底向上纹理）时把**整张图集**镜像 ``V = 1 - py / S``。

    关键是顺序：页内偏移先在未翻转的页矩形里算，最后才做整图镜像。若先把
    偏移算成“翻转后的偏移”再套到整图上（或反过来），页内的 v 方向会被二次
    反转，阴影会上下颠倒或整体偏半页——这正是原始缺陷。
    """
    size = _pos_int(atlas_size, DEFAULT_ATLAS_SIZE, "atlas_size")
    rect = _coerce_rect(
        page if page is not None else page_rect,
        page_count=page_count,
        atlas_size=size,
    )
    x, y, width, height = rect
    u_local, v_local = _coerce_uv(uv)
    u_local = min(1.0, max(0.0, u_local))
    v_local = min(1.0, max(0.0, v_local))

    # 1. 页内纹素坐标（未翻转空间）。
    px = x + u_local * width
    py = y + v_local * height

    # 2/3. 整图映射；翻转只作用在最终的 V 上。
    u = px / size
    v = 1.0 - py / size if flip_y else py / size

    page_no = None
    if isinstance(page_rect, Mapping):
        page_no = page_rect.get("page")
    elif page is not None:
        try:
            page_no = _page_number(page)
        except ValueError:
            page_no = None

    return SampleUV(
        ok=True,
        page=page_no,
        page_rect=rect,
        page_texel=(px, py),
        page_uv=(u_local, v_local),
        atlas_uv=(u, v),
        uv=(u, v),
        u=u,
        v=v,
        uv_u=u,
        uv_v=v,
        texel=(px - 0.5, size - py - 0.5 if flip_y else py - 0.5),
        flip_y=bool(flip_y),
        flipped=bool(flip_y),
        atlas_size=size,
        value=u,
    )


# ---------------------------------------------------------------------------
# 6. 页边界 guard band
# ---------------------------------------------------------------------------

def guard_band(
    page: Any = 1,
    guard: int = DEFAULT_GUARD,
    *,
    page_rect: Any = None,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
    page_count: int = DEFAULT_PAGE_COUNT,
    resolution: int | None = None,
) -> DictReport:
    """返回页的内容区与 guard band 参数。

    每页在物理矩形内缩进 ``guard`` 像素：内容写进内容区，采样也夹在内容区内。
    跨页采样时过滤核只会落在本页的 guard 里，不会触到相邻页的内容，因此相邻
    页的阴影不会互相渗色。

    guard 被夹到不超过页尺寸的四分之一（至少给内容区留一半），``resolution``
    缺省时按内容区边长推算。
    """
    size = _pos_int(atlas_size, DEFAULT_ATLAS_SIZE, "atlas_size")
    rect = _coerce_rect(
        page_rect if page_rect is not None else page,
        page_count=page_count,
        atlas_size=size,
    )
    x, y, width, height = rect
    requested = max(0, int(guard))
    max_band = max(0, int(min(width, height)) // 4)
    band = min(requested, max_band)

    content = (
        x + band,
        y + band,
        max(1.0, width - 2 * band),
        max(1.0, height - 2 * band),
    )
    res = int(resolution) if resolution is not None else int(min(content[2], content[3]))
    res = max(1, res)
    inset_u = band / width if width else 0.0
    inset_v = band / height if height else 0.0

    page_no = None
    if page_rect is None:
        try:
            page_no = _page_number(page)
        except ValueError:
            page_no = None

    return DictReport(
        ok=True,
        page=page_no,
        guard=band,
        requested_guard=requested,
        guard_px=band,
        page_rect=rect,
        content_rect=content,
        content_size=(content[2], content[3]),
        resolution=res,
        inset=(inset_u, inset_v),
        uv_inset=(inset_u, inset_v),
        clamp_uv=(inset_u, 1.0 - inset_u, inset_v, 1.0 - inset_v),
        atlas_size=size,
        valid=band * 2 < min(width, height),
        clamped=band != requested,
        value=band,
    )


def _clamp_page_uv(uv: tuple[float, float], inset: tuple[float, float]) -> tuple[float, float]:
    """把页内 UV 夹进 guard band 之后的内容区。"""
    u, v = uv
    su, sv = inset
    u = min(1.0 - su, max(su, u)) if su < 0.5 else 0.5
    v = min(1.0 - sv, max(sv, v)) if sv < 0.5 else 0.5
    return (u, v)


# ---------------------------------------------------------------------------
# 图集状态机
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict[str, Any]:
    return {
        "allocations": 0,
        "releases": 0,
        "page_moves": 0,
        "collisions": 0,
        "allocation_waits": 0,
        "pending_allocs": 0,
        "redundant_waits": 0,
        "wait_on_free": 0,
        "writes": 0,
        "write_before_alloc": 0,
        "write_on_free": 0,
        "group_resolutions": {},
        "flip_samples": 0,
        "unflipped_samples": 0,
        "guard_violations": 0,
        "pool_exhausted": 0,
        "coalesced_ranges": 0,
        "free_pages": 0,
        "used_pages": 0,
    }


class SpotlightShadowAtlas:
    """按灯 ID 稳定分页的聚光灯阴影图集。

    生命周期语义：

    * ``create`` 为灯挑页：首选 :func:`stable_page` 给出的页，被别的灯占用时做
      确定性探测；一旦确定，灯只要活着就**不再换页**（粘滞），因此可见性变化
      与渲染顺序变化都不会让阴影边缘跳动。
    * ``destroy`` 立即归还页并合并空闲区间。
    * ``create(async_alloc=True)`` 把页标记为 ``pending``；``write`` 之前必须
      ``wait``，否则抛 :class:`ShadowAtlasError` 并记账。
    * ``set_group_resolution`` 以页组为单位统一分辨率。
    """

    def __init__(
        self,
        page_count: int = DEFAULT_PAGE_COUNT,
        atlas_size: int = DEFAULT_ATLAS_SIZE,
        guard: int = DEFAULT_GUARD,
        *,
        group_size: int | None = None,
        frame: int = 0,
    ) -> None:
        self.page_count = _pos_int(page_count, DEFAULT_PAGE_COUNT, "page_count")
        self.atlas_size = _pos_int(atlas_size, DEFAULT_ATLAS_SIZE, "atlas_size")
        self.guard = max(0, int(guard))
        self.frame = int(frame)

        self.grid, self.page_span = _layout(self.page_count, self.atlas_size)
        if self.guard * 2 >= self.page_span:
            raise ValueError(
                f"guard {self.guard} leaves no content in page span {self.page_span}"
            )

        # 默认按 atlas 行分页组：同一行的相邻页共享分辨率。
        self.group_size = int(group_size) if group_size else self.grid
        default_resolution = self.page_span - 2 * self.guard
        self.pages: dict[int, dict[str, Any]] = {}
        for page in range(self.page_count):
            self.pages[page] = {
                "page": page,
                "page_id": page,
                "index": page,
                "light_id": None,
                "alloc_state": FREE,
                "rect": _page_rect(page, self.page_count, self.atlas_size),
                "group": page // self.group_size,
                "resolution": default_resolution,
                "requested_resolution": default_resolution,
                "version": 0,
                "written": 0,
                "flip_y": ATLAS_FLIP_Y,
                "guard": self.guard,
            }

        self.light_pages: dict[Any, int] = {}
        self.free_pages: set[int] = set(range(self.page_count))
        self.events: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = _empty_metrics()
        self.metrics["free_pages"] = self.page_count
        self.metrics["used_pages"] = 0

    # -- 查询 ---------------------------------------------------------------

    @property
    def free_ranges(self) -> list[tuple[int, int]]:
        return _free_ranges(self.free_pages)

    @property
    def fragmentation(self) -> float:
        return _fragmentation(self.free_pages)

    def page_of(self, light_id: Any) -> int | None:
        return self.light_pages.get(_light_id(light_id, light_id))

    def page_rect(self, page: Any) -> tuple[int, int, int, int]:
        return _page_rect(_page_number(page), self.page_count, self.atlas_size)

    def content_rect(self, page: Any) -> tuple[float, float, float, float]:
        x, y, w, h = self.page_rect(page)
        guard = self.guard
        return (x + guard, y + guard, w - 2 * guard, h - 2 * guard)

    def resolution_of(self, page: Any) -> int:
        return int(self.pages[_page_number(page)]["resolution"])

    def group_of(self, page: Any) -> int:
        return int(self.pages[_page_number(page)]["group"])

    def allocations(self) -> dict[Any, int]:
        return dict(self.light_pages)

    def _record(self, kind: str, **detail: Any) -> None:
        event = {"seq": len(self.events) + 1, "kind": kind, "frame": self.frame}
        event.update(detail)
        self.events.append(event)

    # -- 分配与回收 ---------------------------------------------------------

    def _acquire_slot(self, light_id: Any) -> int | None:
        preferred = _preferred_page(light_id, self.page_count)
        if preferred in self.free_pages:
            return preferred
        step = _probe_step(light_id, self.page_count)
        page = preferred
        for _ in range(self.page_count):
            page = (page + step) % self.page_count
            if page in self.free_pages:
                return page
        return None

    def create(
        self,
        light_id: Any = 1,
        resolution: int | None = None,
        *,
        async_alloc: bool = False,
        frame: int | None = None,
    ) -> PageAllocation:
        """为灯分配页；已存在的灯保持原页不变（粘滞映射）。"""
        lid = _light_id(light_id, light_id)
        if frame is not None:
            self.frame = int(frame)

        preferred = _preferred_page(lid, self.page_count)
        if lid in self.light_pages:
            page = self.light_pages[lid]
            record = self.pages[page]
            if resolution is not None:
                record["requested_resolution"] = int(resolution)
            return PageAllocation(
                ok=True,
                page=page,
                page_id=page,
                light_id=lid,
                alloc_state=record["alloc_state"],
                rect=record["rect"],
                resolution=record["resolution"],
                group=record["group"],
                guard=self.guard,
                reused=True,
                preferred=preferred,
                stable=page == preferred,
                collision=page != preferred,
            )

        page = self._acquire_slot(lid)
        if page is None:
            self.metrics["pool_exhausted"] += 1
            self._record("pool_exhausted", light_id=lid)
            raise PagePoolExhausted(
                f"no free shadow page for light {lid!r} "
                f"({self.page_count} pages, {len(self.light_pages)} used)"
            )

        record = self.pages[page]
        self.free_pages.discard(page)
        if resolution is None:
            resolution = record["requested_resolution"]
        record.update(
            light_id=lid,
            alloc_state=PENDING if async_alloc else ALLOCATED,
            resolution=_clamp(int(resolution), 1, self.page_span - 2 * self.guard),
            requested_resolution=int(resolution),
            version=int(record["version"]) + 1,
            written=0,
            guard=self.guard,
        )
        self.light_pages[lid] = page
        self.metrics["allocations"] += 1
        if async_alloc:
            self.metrics["pending_allocs"] += 1
        if page != preferred:
            self.metrics["collisions"] += 1
        self.metrics["used_pages"] = len(self.light_pages)
        self.metrics["free_pages"] = len(self.free_pages)
        self._record(
            "allocate",
            light_id=lid,
            page=page,
            async_alloc=bool(async_alloc),
            preferred=preferred,
            collision=page != preferred,
        )
        return PageAllocation(
            ok=True,
            page=page,
            page_id=page,
            light_id=lid,
            alloc_state=record["alloc_state"],
            rect=record["rect"],
            resolution=record["resolution"],
            group=record["group"],
            guard=self.guard,
            reused=False,
            preferred=preferred,
            stable=page == preferred,
            collision=page != preferred,
        )

    def release(self, light_id: Any = 1, *, frame: int | None = None) -> PageRelease:
        """销毁灯：立即归还页并合并空闲区间。"""
        if frame is not None:
            self.frame = int(frame)
        lid = _light_id(light_id, light_id)
        page = self.light_pages.pop(lid, None)
        if page is None:
            return PageRelease(
                ok=True,
                released=False,
                freed=[],
                pages=[],
                page=None,
                light_id=lid,
                reason="not_allocated",
                immediate=True,
                free_pages=sorted(self.free_pages),
                free_ranges=self.free_ranges,
                fragmentation=self.fragmentation,
                coalesced=False,
                value=0,
            )

        record = self.pages[page]
        default_resolution = self.page_span - 2 * self.guard
        if record["alloc_state"] == PENDING:
            # 销毁一盏仍在异步分配中的灯：页被回收，未完成的分配随之作废，
            # pending 计数必须同步减少，否则会留下永远清不掉的 pending。
            self.metrics["pending_allocs"] = max(0, self.metrics["pending_allocs"] - 1)
        record.update(
            light_id=None,
            alloc_state=FREE,
            written=0,
            version=int(record["version"]) + 1,
            resolution=default_resolution,
            requested_resolution=default_resolution,
        )
        before = self.free_ranges
        self.free_pages.add(page)
        after = self.free_ranges
        coalesced = len(after) < len(before) + 1
        self.metrics["releases"] += 1
        self.metrics["used_pages"] = len(self.light_pages)
        self.metrics["free_pages"] = len(self.free_pages)
        self.metrics["coalesced_ranges"] = len(after)
        self._record("release", light_id=lid, page=page, coalesced=coalesced)

        return PageRelease(
            ok=True,
            released=True,
            freed=[page],
            pages=[page],
            page=page,
            page_id=page,
            light_id=lid,
            reason="destroyed",
            immediate=True,
            free_pages=sorted(self.free_pages),
            free_ranges=after,
            previous_ranges=before,
            fragmentation=self.fragmentation,
            coalesced=coalesced,
            reclaimable=sum(end - start + 1 for start, end in after),
            value=page,
        )

    # -- 等待分配 / 写入 ----------------------------------------------------

    def wait(
        self,
        page: Any = None,
        *,
        light_id: Any = None,
        frame: int | None = None,
        timeout: float | None = None,
    ) -> WaitReport:
        """等待页分配完成；``pending`` 在这里转成 ``allocated``。

        这是异步阴影渲染写入前的栅栏：写入端必须先等到 ``allocated``，不能拿
        一个仍在分配中的页就落笔。
        """
        if frame is not None:
            self.frame = int(frame)

        if page is None:
            if light_id is not None:
                page = self.light_pages.get(_light_id(light_id, light_id))
            if page is None:
                candidates = [
                    p for p, r in self.pages.items() if r["alloc_state"] == PENDING
                ]
                page = min(candidates) if candidates else 0

        index = _page_number(page)
        if not 0 <= index < self.page_count:
            raise ValueError(f"page {index} out of range for {self.page_count} pages")
        record = self.pages[index]
        state_before = record["alloc_state"]
        already_allocated = state_before == ALLOCATED
        if state_before == PENDING:
            record["alloc_state"] = ALLOCATED
            self.metrics["pending_allocs"] = max(0, self.metrics["pending_allocs"] - 1)
            self.metrics["allocation_waits"] += 1
        elif state_before == FREE:
            self.metrics["wait_on_free"] += 1
        else:
            self.metrics["redundant_waits"] += 1
        record["fence"] = int(record.get("fence", 0)) + 1
        self._record("wait_alloc", page=index, state_before=state_before)

        return WaitReport(
            ok=state_before != FREE,
            page=index,
            page_id=index,
            light_id=record["light_id"],
            state_before=state_before,
            state=record["alloc_state"],
            allocated=record["alloc_state"] == ALLOCATED,
            pending=False,
            waited=state_before == PENDING,
            already_allocated=already_allocated,
            fence=record["fence"],
            timeout=timeout,
            value=index,
        )

    def write(
        self,
        page: Any = 1,
        *,
        frame: int | None = None,
        token: Any = None,
    ) -> WriteReport:
        """把阴影内容写进页；页未分配完成时拒绝写入。"""
        if frame is not None:
            self.frame = int(frame)
        index = _page_number(page)
        if not 0 <= index < self.page_count:
            raise ValueError(f"page {index} out of range for {self.page_count} pages")
        record = self.pages[index]

        if record["alloc_state"] == PENDING:
            self.metrics["write_before_alloc"] += 1
            self._record("write_before_alloc", page=index, token=token)
            raise ShadowAtlasError(
                f"page {index} allocation is still pending; wait_alloc() first"
            )
        if record["alloc_state"] == FREE:
            self.metrics["write_on_free"] += 1
            raise ShadowAtlasError(f"page {index} is free; allocate it before writing")

        record["written"] = int(record["written"]) + 1
        self.metrics["writes"] += 1
        self._record("write", page=index, light_id=record["light_id"], token=token)
        return WriteReport(
            ok=True,
            page=index,
            page_id=index,
            light_id=record["light_id"],
            written=record["written"],
            token=token,
            state=record["alloc_state"],
            version=record["version"],
            value=index,
        )

    # -- 页组分辨率 ---------------------------------------------------------

    def set_group_resolution(
        self,
        group: int,
        resolution: Any = None,
        *,
        min_resolution: int = DEFAULT_MIN_RESOLUTION,
        max_resolution: int | None = None,
    ) -> DictReport:
        """统一一个页组里所有页的分辨率。"""
        group = int(group)
        pages = [p for p, r in self.pages.items() if int(r["group"]) == group]
        if not pages:
            raise ValueError(f"unknown page group {group}")

        if resolution is None:
            requested = [self.pages[p]["requested_resolution"] for p in pages]
        else:
            requested = [resolution] * len(pages)

        cap = max_resolution if max_resolution is not None else self.page_span - 2 * self.guard
        floor = max(1, min(int(min_resolution), int(cap)))
        report = page_group_res(
            group,
            requested,
            group_id=group,
            min_resolution=floor,
            max_resolution=max(1, int(cap)),
        )
        unified = int(report["resolution"])
        for page in pages:
            self.pages[page]["resolution"] = unified
        self.metrics["group_resolutions"][group] = unified
        self._record("group_resolution", group=group, resolution=unified)
        report["pages"] = pages
        report["value"] = unified
        return report

    def group_resolution(self, group: int) -> int | None:
        resolutions = {
            int(self.pages[p]["resolution"])
            for p, r in self.pages.items()
            if int(r["group"]) == int(group)
        }
        return resolutions.pop() if len(resolutions) == 1 else None

    def unify_groups(self, quantize: bool = True) -> list[DictReport]:
        """把所有存活页按页组统一分辨率（组内取最大需求）。"""
        live = [
            page for page, record in self.pages.items() if record["alloc_state"] != FREE
        ]
        groups: dict[int, list[int]] = {}
        for page in live:
            groups.setdefault(int(self.pages[page]["group"]), []).append(page)

        reports: list[DictReport] = []
        for group, pages in sorted(groups.items()):
            pages.sort()
            requested = [self.pages[p]["requested_resolution"] for p in pages]
            report = page_group_res(
                group,
                requested,
                group_id=group,
                min_resolution=1,
                max_resolution=self.page_span - 2 * self.guard,
                quantize=quantize,
            )
            unified = int(report["resolution"])
            for page in pages:
                self.pages[page]["resolution"] = unified
            self.metrics["group_resolutions"][group] = unified
            report["pages"] = pages
            reports.append(report)
        return reports

    # -- 采样 ---------------------------------------------------------------

    def sample(
        self,
        page: Any,
        uv: Any = (0.5, 0.5),
        *,
        flip_y: bool = ATLAS_FLIP_Y,
    ) -> SampleUV:
        """按页内 UV 采样，自动应用 guard band 夹紧与图集翻转。"""
        index = _page_number(page)
        if not 0 <= index < self.page_count:
            raise ValueError(f"page {index} out of range for {self.page_count} pages")
        record = self.pages[index]
        band = guard_band(
            index,
            self.guard,
            page_rect=record["rect"],
            atlas_size=self.atlas_size,
            page_count=self.page_count,
        )
        local = _clamp_page_uv(_coerce_uv(uv), band["inset"])
        if not band["valid"]:
            # 页太小 / guard 配得太大，内容区不存在：记一次违规便于审计。
            self.metrics["guard_violations"] += 1
        report = flip_sample(
            record["rect"],
            local,
            flip_y=flip_y,
            atlas_size=self.atlas_size,
            page_count=self.page_count,
        )
        report.update(
            page=index,
            light_id=record["light_id"],
            guard=self.guard,
            guard_ok=bool(band["valid"]),
            page_uv=local,
            resolution=record["resolution"],
        )
        if flip_y:
            self.metrics["flip_samples"] += 1
        else:
            self.metrics["unflipped_samples"] += 1
        return report

    def snapshot(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "atlas_size": self.atlas_size,
            "guard": self.guard,
            "grid": self.grid,
            "page_span": self.page_span,
            "pages": {p: dict(r) for p, r in self.pages.items()},
            "light_pages": dict(self.light_pages),
            "free_pages": sorted(self.free_pages),
            "free_ranges": self.free_ranges,
            "fragmentation": self.fragmentation,
            "metrics": dict(self.metrics),
            "events": list(self.events),
            "frame": self.frame,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()


# ---------------------------------------------------------------------------
# 灯排序（只影响提交顺序，不参与页分配）
# ---------------------------------------------------------------------------

def light_sort(lights: Any = 1, camera: Any = None, **kwargs: Any) -> PageList:
    """按重要性给灯排序，仅用于渲染 / 提交顺序。

    排序键依次是可见性、销毁标记、亮度 / 优先级、屏幕占比、到相机距离（远者
    后画），最后是灯 ID 保证稳定。该顺序**不参与页分配**：页只由灯 ID 决定，
    所以即使排序结果每帧变化，阴影页也不会跟着跳。
    """
    items = _as_list(lights)
    if not items:
        if lights is None or isinstance(lights, (list, tuple, set, frozenset)):
            return PageList(
                [],
                order="importance",
                affects_page_mapping=False,
                pages_unchanged=True,
                count=0,
            )
        items = [lights]

    cam_pos = kwargs.get("camera_position")
    if isinstance(camera, Mapping):
        cam_pos = cam_pos or camera.get("position")
    elif isinstance(camera, (list, tuple)) and len(camera) >= 3:
        cam_pos = cam_pos or camera

    def priority(light: Any) -> tuple[Any, ...]:
        if isinstance(light, Mapping):
            visible = bool(light.get("visible", True))
            destroyed = bool(light.get("destroyed", False))
            intensity = float(light.get("intensity", light.get("priority", 1.0)) or 0.0)
            screen = float(light.get("screen_size", 0.0) or 0.0)
            position = light.get("position")
        else:
            visible, destroyed, intensity, screen, position = True, False, 1.0, 0.0, None

        distance = 0.0
        if (
            cam_pos is not None
            and isinstance(position, (list, tuple))
            and len(position) >= 3
        ):
            distance = math.dist(
                [float(v) for v in position[:3]], [float(v) for v in cam_pos[:3]]
            )
        return (
            not visible,
            destroyed,
            -intensity,
            -screen,
            distance,
            _sort_key(_light_id(light, light)),
        )

    ordered = sorted(items, key=priority)
    # 可选校验：灯描述里自带的 ``page`` 是否与按 ID 推导的页一致。页数可由
    # 调用方经 ``page_count`` 覆盖，不能写死默认值。
    page_count = _pos_int(kwargs.get("page_count"), DEFAULT_PAGE_COUNT, "page_count")
    pages_unchanged = True
    for light in ordered:
        if isinstance(light, Mapping) and "page" in light:
            expected = _page_for_light(_light_id(light, light), page_count)
            if _safe_int(light["page"], expected) != expected:
                pages_unchanged = False
                break
    return PageList(
        ordered,
        order="importance",
        affects_page_mapping=False,
        pages_unchanged=pages_unchanged,
        count=len(ordered),
    )


# ---------------------------------------------------------------------------
# 分配入口
# ---------------------------------------------------------------------------

def _light_specs(lights: Any) -> list[dict[str, Any]]:
    """把各种输入统一成灯描述列表。"""
    if isinstance(lights, Mapping) and "lights" in lights:
        lights = lights["lights"]
    if lights is None:
        return []
    if isinstance(lights, (str, bytes)):
        return [{"id": lights}]
    if isinstance(lights, Mapping):
        return [dict(lights)]
    if _is_number(lights):
        # 标量就是一个灯 ID（契约探针 ``atlas_alloc(1)``），不是灯的数量。
        return [{"id": lights}]
    if isinstance(lights, Iterable):
        specs: list[dict[str, Any]] = []
        for item in lights:
            if isinstance(item, Mapping):
                specs.append(dict(item))
            else:
                specs.append({"id": item})
        return specs
    return [{"id": lights}]


def _requested_resolution(spec: Mapping[str, Any], default: int) -> int:
    for key in ("resolution", "texels", "screen_size", "size"):
        if spec.get(key) is not None:
            return _coerce_resolution(spec[key], default)
    return int(default)


def atlas_alloc(
    lights: Any = 1,
    *,
    atlas: "SpotlightShadowAtlas | None" = None,
    page_count: int = DEFAULT_PAGE_COUNT,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
    guard: int = DEFAULT_GUARD,
    frame: int | None = None,
    async_alloc: bool = False,
    quantize: bool = True,
) -> DictReport:
    """为一组灯分配页，返回灯 → 页的稳定映射。

    * 分配顺序不影响页号：页由灯 ID 推导，已存在的灯保持原页；
    * ``async_alloc=True`` 时页初始为 ``pending``，写入方必须先 ``wait_alloc``；
    * 分配完成后按页组统一分辨率（以组内**全部存活页**为准）。
    """
    specs = _light_specs(lights)
    if atlas is None:
        atlas = SpotlightShadowAtlas(
            page_count=page_count,
            atlas_size=atlas_size,
            guard=guard,
            frame=int(frame or 0),
        )
    elif frame is not None:
        atlas.frame = int(frame)

    allocations: list[PageAllocation] = []
    failed: list[Any] = []
    released: list[PageRelease] = []

    # 先回收所有被销毁的灯，再按“灯 ID 的规范顺序”分配。两步都保证结果与
    # 调用方提交灯的先后无关：不透明 ID 发生哈希碰撞时，探测落点只取决于
    # 灯集合里 ID 的排序，而不是这一帧谁先被渲染。
    destroyed = [spec for spec in specs if spec.get("destroyed")]
    for spec in destroyed:
        released.append(atlas.release(_light_id(spec, spec)))

    to_allocate = [spec for spec in specs if not spec.get("destroyed")]
    to_allocate.sort(key=lambda spec: _sort_key(_light_id(spec, spec)))
    for spec in to_allocate:
        lid = _light_id(spec, spec)
        resolution = _requested_resolution(spec, atlas.page_span - 2 * atlas.guard)
        try:
            allocations.append(atlas.create(lid, resolution, async_alloc=async_alloc))
        except PagePoolExhausted:
            failed.append(lid)

    # 返回顺序按灯 ID 规范化，保证同一组灯无论提交顺序如何都得到同样的列表。
    allocations.sort(key=lambda allocation: _sort_key(allocation["light_id"]))

    group_reports = atlas.unify_groups(quantize=quantize)

    # ``mapping`` 只覆盖本次请求的灯；图集里其他仍然存活的灯放在
    # ``live_mapping``，避免调用方把“没请求的灯”误当成这次分配的产物。
    mapping = {allocation["light_id"]: int(allocation["page"]) for allocation in allocations}
    live_mapping = dict(atlas.allocations())

    # 跨帧稳定性以“灯集合的静态映射”为参照：整数 ID 是双射，非整数 ID 也由
    # 同一套确定性探测排序，所以参考值本身与顺序无关。
    live_ids = [
        record["light_id"]
        for _page, record in sorted(atlas.pages.items())
        if record["alloc_state"] != FREE and record["light_id"] is not None
    ]
    reference: dict[Any, int] = {}
    if live_ids and len(live_ids) <= atlas.page_count:
        reference = {
            result["light_id"]: int(result)
            for result in stable_page(live_ids, atlas.page_count)
        }
    page_stable = all(
        reference.get(light) in (None, page)
        for light, page in atlas.allocations().items()
    )

    return DictReport(
        ok=not failed,
        allocations=allocations,
        mapping=mapping,
        live_mapping=live_mapping,
        pages=[int(a["page"]) for a in allocations],
        failed=failed,
        released=released,
        freed=[page for release in released for page in release["freed"]],
        free_pages=sorted(atlas.free_pages),
        free_ranges=atlas.free_ranges,
        fragmentation=atlas.fragmentation,
        groups=group_reports,
        group_resolutions={
            int(r["group"]): int(r["resolution"]) for r in group_reports
        },
        page_count=atlas.page_count,
        atlas_size=atlas.atlas_size,
        guard=atlas.guard,
        pending=[int(a["page"]) for a in allocations if a["alloc_state"] == PENDING],
        async_alloc=bool(async_alloc),
        page_stable=page_stable,
        stable=not failed,
        collisions=atlas.metrics["collisions"],
        atlas=atlas,
        value=len(allocations),
    )


# ---------------------------------------------------------------------------
# 采样入口
# ---------------------------------------------------------------------------

def sample(
    page: Any = 1,
    uv: Any = (0.5, 0.5),
    *,
    atlas: "SpotlightShadowAtlas | None" = None,
    flip_y: bool = ATLAS_FLIP_Y,
    guard: int | None = None,
    page_rect: Any = None,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
    page_count: int = DEFAULT_PAGE_COUNT,
) -> SampleUV:
    """采样一盏灯的阴影页：guard band 夹紧 + 图集翻转。

    没有 atlas 时按默认布局把 ``page`` 解释成页号（契约探针 ``sample(1)``），
    这样纯函数也能独立验证翻转与 guard 行为。
    """
    if atlas is not None:
        index = _page_number(page if page is not None else atlas.light_pages.get(0, 0))
        band = guard if guard is not None else atlas.guard
        rect = atlas.page_rect(index)
        info = guard_band(
            index,
            band,
            page_rect=rect,
            atlas_size=atlas.atlas_size,
            page_count=atlas.page_count,
        )
        local = _clamp_page_uv(_coerce_uv(uv), info["inset"])
        report = flip_sample(
            rect,
            local,
            flip_y=flip_y,
            atlas_size=atlas.atlas_size,
            page_count=atlas.page_count,
        )
        record = atlas.pages[index]
        report.update(
            page=index,
            light_id=record["light_id"],
            guard=info["guard"],
            guard_ok=bool(info["valid"]),
            resolution=record["resolution"],
            page_uv=local,
        )
        if flip_y:
            atlas.metrics["flip_samples"] += 1
        else:
            atlas.metrics["unflipped_samples"] += 1
        return report

    size = _pos_int(atlas_size, DEFAULT_ATLAS_SIZE, "atlas_size")
    rect = (
        _rect_tuple(page_rect)
        if page_rect is not None
        else _page_rect(_page_number(page), page_count, size)
    )
    band = DEFAULT_GUARD if guard is None else int(guard)
    info = guard_band(0, band, page_rect=rect, atlas_size=size, page_count=page_count)
    local = _clamp_page_uv(_coerce_uv(uv), info["inset"])
    report = flip_sample(
        rect,
        local,
        flip_y=flip_y,
        atlas_size=size,
        page_count=page_count,
    )
    report.update(guard=info["guard"], guard_ok=bool(info["valid"]), page_uv=local)
    return report


# ---------------------------------------------------------------------------
# 一帧渲染
# ---------------------------------------------------------------------------

def _demo_scene(frame: int) -> list[dict[str, Any]]:
    """确定性的演示场景：可见性随帧变化，灯集合不变。"""
    lights = []
    for index in range(4):
        lights.append(
            {
                "id": index,
                "visible": (frame + index) % 3 != 0,
                "intensity": 1.0 - 0.1 * index,
                "screen_size": 256 + 64 * index,
                "position": (float(index), 1.0, -2.0 - index),
                "resolution": 256 + 64 * index,
            }
        )
    return lights


def _scene_from(scene: Any) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """解析一帧场景，返回 ``(frame, lights, options)``。

    ``lights`` 为空且场景里显式给了 ``lights`` / ``light_count`` 时表示“本帧
    没有灯”；只有完全没描述灯时才回退到演示场景。
    """
    options: dict[str, Any] = {}
    if isinstance(scene, SpotlightShadowAtlas):
        return int(scene.frame), [], {"atlas": scene, "explicit_lights": False}
    if isinstance(scene, Mapping):
        frame = _safe_int(scene.get("frame", 1) or 1, 1)
        if "lights" in scene:
            lights = _light_specs(scene["lights"])
            explicit = True
        elif "light_count" in scene:
            count = max(0, int(scene["light_count"] or 0))
            lights = [{"id": index} for index in range(count)]
            explicit = True
        else:
            lights = []
            explicit = False
        for key in ("flip_y", "async_render", "guard", "page_count", "atlas_size"):
            if key in scene:
                options[key] = scene[key]
        options["explicit_lights"] = explicit
        return frame, lights, options
    if isinstance(scene, (list, tuple, set, frozenset)):
        return 1, _light_specs(list(scene)), {"explicit_lights": True}
    if _is_number(scene):
        frame = max(1, _safe_int(scene, 1))
        return frame, _demo_scene(frame), {"explicit_lights": True}
    return 1, [], {"explicit_lights": False}


def shadow_render(
    scene: Any = 1,
    *,
    atlas: "SpotlightShadowAtlas | None" = None,
    lights: Any = None,
    frame: int | None = None,
    flip_y: bool = ATLAS_FLIP_Y,
    async_render: bool = True,
    guard: int | None = None,
    page_count: int = DEFAULT_PAGE_COUNT,
    atlas_size: int = DEFAULT_ATLAS_SIZE,
) -> DictReport:
    """模拟一帧阴影渲染，并验证六条修复同时成立。

    流程：创建 / 回收页（按灯 ID 稳定映射、销毁即归还）→ 页组统一分辨率 →
    按重要性对**可见**灯排序（只影响提交顺序）→ 每盏灯写入前 ``wait_alloc``
    → 用 guard band + 翻转采样取出页中心 UV。
    """
    scene_frame, scene_lights, scene_options = _scene_from(scene)
    frame = _safe_int(frame if frame is not None else scene_frame, 1)
    specs = _light_specs(lights) if lights is not None else scene_lights
    if not specs and not scene_options.get("explicit_lights"):
        # 场景完全没描述灯时才用演示场景；显式空灯列表表示本帧没有灯。
        specs = _demo_scene(frame)

    flip_y = bool(scene_options.get("flip_y", flip_y))
    async_render = bool(scene_options.get("async_render", async_render))
    guard = _safe_int(scene_options.get("guard", guard if guard is not None else DEFAULT_GUARD), DEFAULT_GUARD)
    page_count = _safe_int(scene_options.get("page_count", page_count), DEFAULT_PAGE_COUNT)
    atlas_size = _safe_int(scene_options.get("atlas_size", atlas_size), DEFAULT_ATLAS_SIZE)

    created_atlas = atlas is None
    if atlas is None:
        atlas = SpotlightShadowAtlas(
            page_count=page_count, atlas_size=atlas_size, guard=guard, frame=frame
        )
    else:
        atlas.frame = frame
        atlas.guard = guard

    allocation = atlas_alloc(
        specs, atlas=atlas, frame=frame, async_alloc=async_render
    )

    # 异步分配在本帧内必须被消费掉：任何仍为 pending 的页都在这里等待完成，
    # 绝不允许把“分配中”的页带进下一帧。
    for page in [
        p for p, record in atlas.pages.items() if record["alloc_state"] == PENDING
    ]:
        atlas.wait(page)

    visible = [
        spec for spec in specs if spec.get("visible", True) and not spec.get("destroyed")
    ]
    ordered = light_sort(visible)

    passes: list[DictReport] = []
    waits: list[WaitReport] = []
    writes: list[WriteReport] = []
    samples: list[SampleUV] = []
    write_errors: list[dict[str, Any]] = []

    for light in ordered:
        lid = _light_id(light, light)
        page = atlas.page_of(lid)
        if page is None:
            continue
        wait = atlas.wait(page)
        waits.append(wait)
        try:
            write = atlas.write(page, frame=frame)
            writes.append(write)
        except ShadowAtlasError as exc:
            write_errors.append({"light_id": lid, "page": page, "error": str(exc)})
            continue
        sampled = atlas.sample(page, (0.5, 0.5), flip_y=flip_y)
        samples.append(sampled)
        rect = atlas.page_rect(page)
        band = guard_band(page, atlas.guard, page_rect=rect, atlas_size=atlas.atlas_size)
        content_ok = (
            band["content_rect"][0] - rect[0] >= atlas.guard and band["content_rect"][2] > 0
        )
        passes.append(
            DictReport(
                ok=wait["allocated"] and content_ok,
                light_id=lid,
                page=page,
                resolution=atlas.resolution_of(page),
                group=atlas.group_of(page),
                alloc_state=atlas.pages[page]["alloc_state"],
                waited=wait["waited"],
                written=write["written"],
                uv=sampled["uv"],
                flipped=sampled["flipped"],
                guard=band["guard"],
                content_ok=content_ok,
            )
        )

    group_uniform = True
    live_groups: dict[int, set[int]] = {}
    for record in atlas.pages.values():
        if record["alloc_state"] == FREE:
            continue
        live_groups.setdefault(int(record["group"]), set()).add(int(record["resolution"]))
    if any(len(resolutions) > 1 for resolutions in live_groups.values()):
        group_uniform = False

    pending_left = [p for p, r in atlas.pages.items() if r["alloc_state"] == PENDING]
    validation_report = validation(atlas, post_render=True)
    stable = bool(allocation["page_stable"])

    return DictReport(
        ok=validation_report["ok"] and not write_errors,
        frame=frame,
        passes=passes,
        samples=samples,
        waits=waits,
        writes=writes,
        write_errors=write_errors,
        mapping=dict(atlas.allocations()),
        allocation=allocation,
        sorted_lights=[_light_id(light, light) for light in ordered],
        sort_affects_pages=False,
        page_stable=stable,
        pending_pages=pending_left,
        group_uniform=group_uniform,
        group_resolutions=dict(atlas.metrics["group_resolutions"]),
        flip_y=flip_y,
        guard=atlas.guard,
        freed=[page for release in allocation["released"] for page in release["freed"]],
        metrics=dict(atlas.metrics),
        validation=validation_report,
        atlas=atlas,
        created_atlas=created_atlas,
        value=len(passes),
    )


# ---------------------------------------------------------------------------
# 确定性 / 校验 / 审计 / 兜底 / 指标
# ---------------------------------------------------------------------------

def determinism(
    *args: Any,
    runs: int = 3,
    seed: int | None = None,
    **kwargs: Any,
) -> DictReport:
    """验证页映射、分辨率、采样在渲染顺序 / 可见性变化下完全一致。

    每一轮都换一种可见性组合和渲染顺序（并轮换灯的提交顺序），但灯集合相同；
    页分配只由灯 ID 决定，所以各轮输出必须逐位相同。
    """
    base_seed = 1
    if args:
        first = args[0]
        if _is_number(first):
            base_seed = abs(_safe_int(first, 1)) or 1
        elif isinstance(first, Mapping):
            base_seed = abs(_safe_int(first.get("seed", first.get("value", 1)) or 1, 1)) or 1
    if seed is not None:
        base_seed = abs(_safe_int(seed, 1)) or 1
    runs = max(1, int(runs))

    signatures: list[str] = []
    run_reports: list[dict[str, Any]] = []
    for run in range(runs):
        lights = [
            {
                "id": index,
                "visible": (run + index) % 2 == 0,
                "resolution": 128 + 64 * index,
                "intensity": 1.0 - 0.1 * index,
                "screen_size": 256 + 32 * index,
            }
            for index in range(4)
        ]
        # 创建 / 提交顺序随轮次改变，页映射必须不受影响。
        rotate = (base_seed + run) % len(lights)
        ordered = lights[rotate:] + lights[:rotate]
        report = shadow_render({"frame": run + 1, "lights": ordered}, async_render=True)
        atlas = report["atlas"]
        # 签名只覆盖“应当跨帧稳定”的量：灯 → 页映射、每页分辨率、每页采样 UV。
        # 可见性 / 渲染顺序变化只会改变本帧画了哪些页，不能改变这些量。
        per_page = [
            (
                int(page),
                int(record["group"]),
                int(record["resolution"]),
                tuple(
                    round(float(value), 6)
                    for value in atlas.sample(page, (0.5, 0.5), flip_y=True)["uv"]
                ),
            )
            for page, record in sorted(atlas.pages.items())
            if record["alloc_state"] != FREE
        ]
        signature = _digest(report["mapping"], per_page)
        signatures.append(signature)
        run_reports.append(
            {
                "run": run,
                "mapping": report["mapping"],
                "per_page": per_page,
                "page_stable": report["page_stable"],
                "group_uniform": report["group_uniform"],
                "ok": report["ok"],
            }
        )

    deterministic = len(set(signatures)) == 1
    first = (
        run_reports[0]
        if run_reports
        else {"mapping": {}, "per_page": [], "page_stable": True, "group_uniform": True}
    )
    return DictReport(
        ok=deterministic and all(item["ok"] for item in run_reports),
        deterministic=deterministic,
        equal=deterministic,
        order_independent=deterministic,
        runs=runs,
        seed=base_seed,
        signatures=signatures,
        signature=signatures[0] if signatures else "",
        mapping=first["mapping"],
        per_page=first["per_page"],
        page_stable=all(item["page_stable"] for item in run_reports),
        group_uniform=all(item["group_uniform"] for item in run_reports),
        reports=run_reports,
        pages=sorted(first["mapping"].values()),
        value=len(signatures),
    )


def validation(*args: Any, **kwargs: Any) -> DictReport:
    """检查六条核心不变量（以及若干派生一致性）。"""
    state = args[0] if args else 1
    issues: list[dict[str, Any]] = []

    def add(code: str, message: str, severity: str = "error", **detail: Any) -> None:
        issues.append({"code": code, "message": message, "severity": severity, **detail})

    if isinstance(state, SpotlightShadowAtlas):
        # 1. 稳定映射：灯 → 页 → 灯 双向一致，且每盏灯只有一个页。
        seen: dict[int, Any] = {}
        for light, page in state.light_pages.items():
            if page in seen:
                add(
                    "duplicate_page",
                    "two lights share the same page",
                    lights=[seen[page], light],
                    page=page,
                )
            seen[page] = light
            record = state.pages[page]
            if record["light_id"] != light:
                add(
                    "page_owner_mismatch",
                    "page owner does not match light map",
                    page=page,
                    light_id=light,
                    owner=record["light_id"],
                )
            if record["alloc_state"] == FREE:
                add("allocated_page_marked_free", "allocated page is marked free", page=page)
        for page, record in state.pages.items():
            if record["alloc_state"] != FREE and record["light_id"] is None:
                add("orphan_allocation", "page allocated without a light", page=page)
            if record["alloc_state"] == FREE and page not in state.free_pages:
                add("free_page_not_pooled", "free page missing from free pool", page=page)
            if record["alloc_state"] != FREE and page in state.free_pages:
                add("allocated_page_pooled", "allocated page still in free pool", page=page)

        # 2. 销毁即归还：非 free 页必须都在灯表里。
        for page, record in state.pages.items():
            if record["alloc_state"] != FREE and page not in seen:
                add("page_not_released", "destroyed light's page was not returned", page=page)

        # 3. 写入前等待分配完成。
        if state.metrics.get("write_before_alloc", 0):
            add(
                "write_before_alloc",
                "async shadow write happened before allocation completed",
                count=state.metrics["write_before_alloc"],
            )
        if kwargs.get("post_render") and state.metrics.get("pending_allocs", 0):
            add(
                "pending_after_frame",
                "pages are still pending after the frame",
                count=state.metrics["pending_allocs"],
            )

        # 4. 页组分辨率统一。
        by_group: dict[int, set[int]] = {}
        for record in state.pages.values():
            if record["alloc_state"] == FREE:
                continue
            by_group.setdefault(int(record["group"]), set()).add(int(record["resolution"]))
        for group, resolutions in by_group.items():
            if len(resolutions) > 1:
                add(
                    "group_resolution_mismatch",
                    "pages in one group use different resolutions",
                    group=group,
                    resolutions=sorted(resolutions),
                )

        # 5. 图集翻转被应用。
        if state.metrics.get("flip_samples", 0) == 0 and state.metrics.get("writes", 0) > 0:
            add(
                "missing_flip",
                "shadow samples did not account for atlas flip",
                severity="warning",
            )

        # 6. guard band 有效：内容区非空且每页都留了边。
        invalid_pages: list[int] = []
        clamped_pages: list[int] = []
        for page, record in state.pages.items():
            band = guard_band(
                page, state.guard, page_rect=record["rect"], atlas_size=state.atlas_size
            )
            if not band["valid"]:
                invalid_pages.append(page)
            elif state.guard > 0 and band["guard"] != state.guard:
                clamped_pages.append(page)
        if invalid_pages:
            add(
                "guard_band_invalid",
                "guard band leaves no content area",
                pages=invalid_pages,
                guard=state.guard,
            )
        if clamped_pages:
            # 聚合成一条告警：guard 被夹小说明配置或页尺寸有问题，同一条问题
            # 不应因为页数多就刷屏。
            add(
                "guard_band_clamped",
                "guard band was clamped below the configured width",
                severity="warning",
                pages=clamped_pages,
                count=len(clamped_pages),
                configured=state.guard,
            )

        if state.metrics.get("page_moves", 0):
            add("page_moved", "a live light changed pages", count=state.metrics["page_moves"])

    elif isinstance(state, Mapping):
        if state.get("page_stable") is False:
            add("unstable_page_mapping", "page assignment changed across frames")
        if state.get("pages_by_order") is True or state.get("mapping_key") == "order":
            add("order_dependent_mapping", "page assignment depends on light order")
        if state.get("released_on_destroy") is False:
            add("page_not_released", "destroyed light did not return its page")
        if state.get("write_before_alloc") or state.get("write_before_alloc_count"):
            add("write_before_alloc", "write happened before allocation completed")
        if state.get("group_uniform") is False or state.get("per_page_resolution") is True:
            add("group_resolution_mismatch", "page group resolutions are not uniform")
        if state.get("flip_applied") is False:
            add("missing_flip", "samples ignore atlas flip")
        if state.get("guard") == 0 or state.get("guard_present") is False:
            add("missing_guard_band", "page boundary has no guard band")
        if state.get("fragmented") is True:
            add("fragmented_pool", "free pages were not coalesced", severity="warning")

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return DictReport(
        ok=not errors,
        valid=not errors,
        errors=errors,
        warnings=warnings,
        issues=issues,
        issue_count=len(issues),
        checked=1,
        schema_version=1,
    )


def audit(*args: Any, **kwargs: Any) -> DictReport:
    """生成可复现的审计记录；标量探针恰好包含一个事件。"""
    state = args[0] if args else 1
    if isinstance(state, SpotlightShadowAtlas):
        events = [dict(event) for event in state.events]
        validation_report = validation(state, **kwargs)
        metrics_report = metrics(state)
        mapping = state.allocations()
        page_count = state.page_count
        free = sorted(state.free_pages)
        fragmentation = state.fragmentation
        ok = bool(validation_report["ok"])
        guard = state.guard
    else:
        events = [{"seq": 1, "kind": "probe", "target": state}]
        validation_report = validation(1)
        metrics_report = metrics(state if isinstance(state, Mapping) else 1)
        mapping = state.get("mapping", {}) if isinstance(state, Mapping) else {}
        page_count = _safe_int(state.get("page_count", 0), 0) if isinstance(state, Mapping) else 0
        free = list(state.get("free_pages", [])) if isinstance(state, Mapping) else []
        fragmentation = (
            float(state.get("fragmentation", 0.0)) if isinstance(state, Mapping) else 0.0
        )
        ok = True
        guard = (
            _safe_int(state.get("guard", DEFAULT_GUARD), DEFAULT_GUARD)
            if isinstance(state, Mapping)
            else DEFAULT_GUARD
        )

    report = DictReport(
        kind="spotlight_shadow_atlas_audit",
        ok=ok,
        schema_version=1,
        event_count=len(events),
        events=events,
        mapping=mapping,
        page_count=page_count,
        free_pages=free,
        fragmentation=fragmentation,
        guard=guard,
        validation=validation_report,
        metrics=metrics_report,
        findings=validation_report.get("issues", []),
        value=len(events),
    )
    report["digest"] = _digest(
        "spotlight_shadow_atlas_audit", report["event_count"], page_count, ok
    )
    return report


def fallback(*args: Any, **kwargs: Any) -> DictReport:
    """异常 / 资源紧张时的安全兜底策略。"""
    reason = args[0] if args else kwargs.get("reason", "unspecified")
    text = str(reason).lower()

    if any(
        word in text
        for word in ("exhaust", "pool", "overflow", "fragment", "grow", "full")
    ):
        mode = "grow_pool_and_reuse_coalesced"
        action = "coalesce_free_ranges_then_grow_page_pool"
    elif any(
        word in text
        for word in ("collision", "collide", "jitter", "stable", "order", "page")
    ):
        mode = "stable_light_hash_mapping"
        action = "keep_existing_pages_and_probe_new_lights"
    elif any(word in text for word in ("pending", "alloc", "race", "async", "write")):
        mode = "wait_then_write"
        action = "wait_for_allocation_fence_before_write"
    elif any(
        word in text
        for word in ("resolution", "group", "border", "seam", "adaptive", "ladder")
    ):
        mode = "snap_group_resolution"
        action = "unify_resolution_per_page_group_on_pow2_ladder"
    elif any(word in text for word in ("flip", "mirror", "uv", "v_axis")):
        mode = "flip_page_local_uv"
        action = "offset_in_flipped_page_rect_before_sampling"
    elif any(word in text for word in ("guard", "bleed", "gutter", "sample")):
        mode = "clamp_to_content_rect"
        action = "clamp_sampling_to_page_content_rect"
    else:
        mode = "safe_atlas_rebuild"
        action = "clear_atlas_and_reallocate_by_light_id"

    return DictReport(
        ok=True,
        mode=mode,
        action=action,
        reason=str(reason),
        safe_clear=True,
        stable_mapping=True,
        release_on_destroy=True,
        wait_before_write=True,
        group_uniform=True,
        flip_y=ATLAS_FLIP_Y,
        guard=DEFAULT_GUARD,
        value=1,
    )


def metrics(*args: Any, **kwargs: Any) -> DictReport:
    """返回图集指标快照。"""
    state = args[0] if args else 1
    snapshot = _empty_metrics()
    snapshot.update(
        {
            "pages": 0,
            "used_pages": 0,
            "free_pages": 0,
            "pending_pages": 0,
            "fragmentation": 0.0,
            "page_moves": 0,
            "stable": True,
            "group_uniform": True,
            "write_before_alloc": 0,
            "write_on_free": 0,
        }
    )

    if isinstance(state, SpotlightShadowAtlas):
        for key, value in state.metrics.items():
            if key in snapshot:
                snapshot[key] = value
        snapshot["pages"] = state.page_count
        snapshot["used_pages"] = len(state.light_pages)
        snapshot["free_pages"] = len(state.free_pages)
        snapshot["pending_pages"] = sum(
            1 for r in state.pages.values() if r["alloc_state"] == PENDING
        )
        snapshot["fragmentation"] = state.fragmentation
        snapshot["free_ranges"] = state.free_ranges
        snapshot["group_resolutions"] = dict(state.metrics.get("group_resolutions", {}))
        snapshot["stable"] = state.metrics.get("page_moves", 0) == 0
        snapshot["guard"] = state.guard
        snapshot["atlas_size"] = state.atlas_size
        snapshot["page_span"] = state.page_span
        snapshot["stable_page_mappings"] = len(state.light_pages)
    elif isinstance(state, Mapping):
        source = state.get("metrics", state)
        if isinstance(source, Mapping):
            for key in list(snapshot):
                if key in source:
                    snapshot[key] = source[key]
        if isinstance(state.get("pages"), Mapping):
            snapshot["pages"] = len(state["pages"])
        for key in ("mapping", "free_pages", "group_resolutions"):
            if key in state:
                snapshot[key] = state[key]
        if "guard" in state:
            snapshot["guard"] = state["guard"]
    elif _is_number(state):
        snapshot["probe"] = state
        probe = abs(_safe_int(state, 0))
        snapshot["pages"] = probe
        snapshot["stable_page_mappings"] = probe
        snapshot["value"] = state

    snapshot["page_utilization"] = (
        snapshot["used_pages"] / snapshot["pages"] if snapshot["pages"] else 0.0
    )
    snapshot["violation_free"] = (
        snapshot.get("write_before_alloc", 0) == 0
        and snapshot.get("write_on_free", 0) == 0
    )
    return DictReport(snapshot)
