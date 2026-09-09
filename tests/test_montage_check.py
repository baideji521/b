"""混剪输出守门人回归测试。

盯的是一句话：**PRM 里能算的规则，程序必须替人算准，不许放过也不许误报。**

  T1  合规的一条 montage -> 一处不报
  T2  素材低于 85 分（四维和 < 17）-> 报，并把分数换算说清
  T3  duration 没照抄输入 -> 报
  T4  挂字时间没照抄输入 -> 报
  T5  挂字文案没照抄（丢了 emoji / 换了词）-> 报
  T6  输入有挂字却漏了没输出 -> 报
  T7  输入没有挂字却自己造了一条 -> 报
  T8  输出里出现别的 overlay type（emoji 单独成条）-> 报
  T9  Rank 1 的高潮保护区里有旁白 -> 报
  T10 同一组里有两条同源（source 数字 ID 相同）-> 报
  T11 跨 montage 复用同一条素材 -> 不报（PRM 允许）
  T12 末尾硬插起说时间算错（拿 1.2 当万能 TTS 时长）-> 报，并给出应该是多少
  T13 空隙装不下这句话 -> 报，并说清压到了哪
  T14 除 Rank 1 外某段一条 comment 都没有 -> 报
  T15 开场白不在 0.0 / 和 opening.text 不一致 -> 报
  T16 TTS 时长查表：1-5 词 + 句内标点，且不是线性公式
  T17 一条一个对象并排写（含 ```json 围栏、单条、老的 montages 包裹）都要认出来
  T18 整个回复是空的（一组都凑不出来）-> 不当错误，报「没有可检查的」
  T19 评分表 CSV：机器判定填好、主观项留空、公式行号对得上




纯算时间，不碰磁盘、不碰数据库。
可以直接 `python tests/test_montage_check.py`，也可以 `pytest tests/test_montage_check.py`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidscribe.highlight import montage as mg                      # noqa: E402


# ------------------------------------------------------------------ 造数据
#: 输入 `o` 里那条挂字的文案。输出必须一个字符不差地照抄它
WORD = "BOOM 😱"


def pool_row(name: str, *, source: str, duration: float, word_at: float,
             gaps: list[list[float]] | None = None,
             scores: dict[str, int] | None = None) -> dict:
    """一行「提取数据」。默认四维 5/5/5/4 = 19（95 分），稳过门槛。"""
    return {
        "video": name,
        "source": source,
        "timeline": {
            "duration": duration,
            "scores": scores or {"surprise": 5, "standalone": 5,
                                 "emotion_power": 5, "caption": 4},
            "type": "测试",
        },
        "o": [[word_at, "word", WORD]],
        "t": {"Scene": "室内", "Action": "看结果", "Speech text": "whatever"},
        "gaps": gaps or [],
        "startframe": 0.0,
        "freeze": 0.0,
    }



def base_pool() -> dict[str, dict]:
    """5 条素材，5 个不同 ID。第 1 条（play_order 首位）留给开场白。"""
    rows = [
        pool_row("A_1.mp4", source="a_tiktok_1111.mp4", duration=4.02, word_at=3.04),
        pool_row("B_2.mp4", source="b_tiktok_2222.mp4", duration=7.11, word_at=4.10),
        pool_row("C_3.mp4", source="c_tiktok_3333.mp4", duration=8.52, word_at=5.08,
                 gaps=[[5.94, 6.93]]),
        pool_row("D_4.mp4", source="d_tiktok_4444.mp4", duration=6.72, word_at=2.20),
        pool_row("E_5.mp4", source="e_tiktok_5555.mp4", duration=6.89, word_at=2.66),
    ]
    return {one["video"]: one for one in rows}


OPENING = "These keep getting crazier."


def base_montage() -> dict:
    """一条合规的 montage：开场白在 0.0，一条走空隙，两条末尾硬插，Rank 1 闭嘴。

    每一条的 `word` 都是从输入的 `o` **原样照抄**（时间 + 文案）。
    时间怎么来的：
      C_3  gaps [[5.94, 6.93]] 窗口 0.99 -> 可容 0.69 -> 1 词 "Brutal."(0.60)
           起说 5.94 + 0.15 = 6.09
      B_2  gaps [] -> 末尾硬插 "Do the math."(3 词 = 1.20) -> 7.11 - 1.20 = 5.91
      D_4  gaps [] -> 末尾硬插 "Wow."(1 词 = 0.60)         -> 6.72 - 0.60 = 6.12
      E_5  Rank 1，word 2.66 -> 保护区 [1.66, 6.89]，只有照抄的 word，没有 comment
    """
    return {
        "opening": {"text": OPENING},
        "selected": [
            {"rank": 1, "duration": 6.89, "video": "E_5.mp4",
             "overlays": [[2.66, "word", WORD]]},
            {"rank": 2, "duration": 8.52, "video": "C_3.mp4",
             "overlays": [[5.08, "word", WORD], [6.09, "comment", "Brutal."]]},
            {"rank": 3, "duration": 7.11, "video": "B_2.mp4",
             "overlays": [[4.10, "word", WORD], [5.91, "comment", "Do the math."]]},
            {"rank": 4, "duration": 6.72, "video": "D_4.mp4",
             "overlays": [[2.20, "word", WORD], [6.12, "comment", "Wow."]]},
            {"rank": 5, "duration": 4.02, "video": "A_1.mp4",
             "overlays": [[0.0, "comment", OPENING], [3.04, "word", WORD]]},
        ],
        "play_order": ["A_1.mp4", "D_4.mp4", "B_2.mp4", "C_3.mp4", "E_5.mp4"],
    }




def only(bad: list[str], needle: str) -> list[str]:
    return [one for one in bad if needle in one]


def set_comment(pick: dict, at: float, text: str) -> None:
    """把这一段的 comment 换成 `[at, "comment", text]`，挂字原样不动。

    按 type 找，不按下标 —— 下标会随 overlays 的排列变，改一次数据就得改一遍测试。
    """
    keep = [one for one in pick["overlays"] if str(one[1]) != "comment"]
    pick["overlays"] = keep + [[at, "comment", text]]



# ------------------------------------------------------------------ 用例
def test_clean_montage_reports_nothing() -> None:
    bad = mg.check_montage(base_montage(), base_pool())
    assert bad == [], "合规的一条不该报任何问题：%s" % bad


def test_low_score_material_is_rejected() -> None:
    pool = base_pool()
    pool["C_3.mp4"]["timeline"]["scores"] = {"surprise": 4, "standalone": 4,
                                             "emotion_power": 4, "caption": 4}  # 16 = 80 分
    bad = mg.check_montage(base_montage(), pool)
    hit = only(bad, "16/20")
    assert hit, "80 分的素材必须报出来：%s" % bad
    assert "80.0" in hit[0], "要把百分制换算说清：%s" % hit[0]


def test_duration_must_be_copied() -> None:
    one = base_montage()
    one["selected"][1]["duration"] = 8.5          # 输入是 8.52
    bad = mg.check_montage(one, base_pool())
    assert only(bad, "duration"), "改了 duration 必须报：%s" % bad


def test_word_time_must_be_copied() -> None:
    one = base_montage()
    one["selected"][1]["overlays"][0][0] = 5.0     # 输入是 5.08
    bad = mg.check_montage(one, base_pool())
    hit = only(bad, "挂字时间")
    assert hit, "改了挂字时间必须报：%s" % bad
    assert "5.08" in hit[0], "要给出输入的原值：%s" % hit[0]


def test_word_text_must_be_copied_verbatim() -> None:
    one = base_montage()
    one["selected"][1]["overlays"][0][2] = "BOOM"   # emoji 被弄丢了
    bad = mg.check_montage(one, base_pool())
    hit = only(bad, "挂字文案")
    assert hit, "改了挂字文案必须报：%s" % bad
    assert "不许润色" in hit[0], "要说清不许自己想：%s" % hit[0]


def test_missing_word_is_reported() -> None:
    one = base_montage()
    one["selected"][1]["overlays"] = [[6.09, "comment", "Brutal."]]
    bad = mg.check_montage(one, base_pool())
    assert only(bad, "漏了挂字"), "输入有 word 却没照抄出来，必须报：%s" % bad


def test_self_invented_word_is_reported() -> None:
    pool = base_pool()
    pool["C_3.mp4"]["o"] = []                       # 输入压根没有挂字
    bad = mg.check_montage(base_montage(), pool)
    assert only(bad, "不许自己造"), "输入没有挂字却写了一条，必须报：%s" % bad


def test_unknown_overlay_kind_is_rejected() -> None:
    one = base_montage()
    one["selected"][1]["overlays"].insert(0, [3.0, "emoji", "😱"])
    bad = mg.check_montage(one, base_pool())
    hit = only(bad, "emoji")
    assert hit, "emoji 不许单独成条：%s" % bad
    assert "写在 word 的文案里" in hit[0], "要说清 emoji 该在哪：%s" % hit[0]


def test_rank1_climax_guard_is_hard() -> None:
    one = base_montage()
    # E_5 是 Rank 1，word 在 2.66 -> 保护区从 1.66 开始
    one["selected"][0]["overlays"].append([2.00, "comment", "Wow."])
    bad = mg.check_montage(one, base_pool())
    assert only(bad, "高潮保护区"), "Rank 1 高潮区有旁白必须报：%s" % bad




def test_same_source_id_in_one_montage_is_rejected() -> None:
    pool = base_pool()
    # C 和 D 换成同一条原片下载两次：source 字符串不同，数字 ID 相同
    pool["C_3.mp4"]["source"] = "202609081002_tiktok_9999.mp4"
    pool["D_4.mp4"]["source"] = "202609080351_tiktok_9999.mp4"
    bad = mg.check_montage(base_montage(), pool)
    hit = only(bad, "同源撞车")
    assert hit, "同一组里同 ID 必须报：%s" % bad
    assert "9999" in hit[0], "要报出是哪个 ID：%s" % hit[0]


def test_reuse_across_montages_is_allowed() -> None:
    pool = base_pool()
    payload = {"montages": [base_montage(), base_montage()]}
    checked = mg.check_all(payload, pool)
    assert len(checked) == 2, "两条都要检查：%s" % checked
    for index, bad in checked:
        assert bad == [], "跨 montage 复用是允许的，第 %d 条不该报：%s" % (index, bad)


def test_tail_insert_start_must_be_computed_not_guessed() -> None:
    one = base_montage()
    # 拿 1.2 当万能 TTS 时长：7.11 - 1.2 = 5.91 恰好和 3 词的正确值撞上，
    # 所以换一句 4 词的来验 —— 4 词要 1.60，正确起说是 7.11 - 1.60 = 5.51
    set_comment(one["selected"][2], 5.91, "Four years is brutal.")


    bad = mg.check_montage(one, base_pool())
    hit = only(bad, "末尾硬插")
    assert hit, "末尾硬插算错必须报：%s" % bad
    assert "5.51" in hit[0], "要给出应该是多少：%s" % hit[0]
    assert only(bad, "说不完"), "超出素材时长也要单独报：%s" % bad


def test_gap_too_small_is_reported() -> None:
    one = base_montage()
    # C_3 的空隙只有 0.99 秒（可容 0.69），塞一句 4 词（1.60）进去
    set_comment(one["selected"][1], 6.09, "He needs to see.")


    bad = mg.check_montage(one, base_pool())
    hit = only(bad, "容得下")
    assert hit, "空隙装不下必须报：%s" % bad
    assert "6.93" in hit[0], "要说清原声什么时候回来：%s" % hit[0]


def test_every_pick_but_rank1_needs_a_comment() -> None:

    one = base_montage()
    one["selected"][3]["overlays"] = [[2.20, "word", WORD]]   # rank4 的旁白删掉

    bad = mg.check_montage(one, base_pool())
    assert only(bad, "一条 comment 都没有"), "缺旁白必须报：%s" % bad
    assert only(bad, "下限"), "总数不够也要报：%s" % bad


def test_opening_must_match_and_sit_at_zero() -> None:
    one = base_montage()
    one["selected"][4]["overlays"][0][2] = "Something else."
    bad = mg.check_montage(one, base_pool())
    assert only(bad, "opening.text"), "开场白文案不一致必须报：%s" % bad

    two = base_montage()
    two["selected"][4]["overlays"][0][0] = 0.5    # 挪走 -> 不再是开场白，按落位规则判
    bad = mg.check_montage(two, base_pool())
    assert bad, "开场白不在 0.0 必须报：%s" % bad


def test_tts_table_is_not_linear() -> None:
    assert mg.tts_seconds("Wow.") == 0.60
    assert mg.tts_seconds("Do the math.") == 1.20
    assert mg.tts_seconds("Four years is brutal.") == 1.60
    assert mg.tts_seconds("He is definitely cooked now.") == 1.90
    # 句内逗号加 0.15，结尾句号不加
    assert mg.tts_seconds("Wait, do the math.") == 1.75
    # 不是线性：1 词到 5 词不成正比（线性会给 5 词 3.00）
    assert mg.tts_seconds("He is definitely cooked now.") < 5 * mg.tts_seconds("Wow.")
    assert mg.word_count("BOOM 😱") == 1, "emoji 不算词"


def test_side_by_side_objects_are_recognised() -> None:
    """一条一个对象并排写（现在要求的写法），要能全部认出来。"""
    one, two = base_montage(), base_montage()
    plain = json.dumps(one, ensure_ascii=False) + "\n\n" + json.dumps(two, ensure_ascii=False)
    assert len(mg.montages_of(plain)) == 2, "并排的两个对象要认成两条"

    fenced = "```json\n" + plain + "\n```"
    assert len(mg.montages_of(fenced)) == 2, "带 ```json 围栏也要认"

    single = json.dumps(one, ensure_ascii=False)
    assert len(mg.montages_of(single)) == 1, "只有一条时就是一个对象"

    wrapped = json.dumps({"montages": [one, two]}, ensure_ascii=False)
    assert len(mg.montages_of(wrapped)) == 2, "老写法 montages 照旧要认"

    checked = mg.check_all(plain, base_pool())
    assert len(checked) == 2
    for index, bad in checked:
        assert bad == [], "并排写法不该因为形状被判违规，第 %d 条：%s" % (index, bad)


def test_empty_reply_is_not_an_error() -> None:

    assert mg.montages_of("") == []
    assert mg.montages_of("   \n ") == []
    checked = mg.check_all("", base_pool())
    assert checked == []
    assert "没有可检查" in mg.report(checked)


def test_scorecard_fills_machine_part_and_leaves_manual_blank() -> None:
    pool = base_pool()
    one = base_montage()
    two = base_montage()
    two["selected"][1]["duration"] = 8.5          # 第二条故意不合规
    payload = {"montages": [one, two]}
    checked = mg.check_all(payload, pool)
    csv = mg.scorecard(checked, payload, pool)
    rows = [line.split(",") for line in csv.splitlines()]

    assert rows[0][0] == "维度 / 评分项"
    assert rows[0][2] == "montage1" and rows[0][3] == "montage2"
    assert rows[1][2] == "通过", "第一条应判通过：%s" % rows[1]
    assert rows[1][3].startswith("淘汰"), "第二条应判淘汰：%s" % rows[1]

    names = [row[0] for row in rows]
    for item, _full in mg.SCORE_ITEMS:
        assert item in names, "主观项 %s 应该在表里" % item
    assert "故事 · 主题收敛" not in names, "无区分度的项应该已经砍掉"
    assert "节奏 · 旁白空隙可用" not in names, "机器已经算过的项不该再让人打分"
    assert "情绪 · 高潮前静默" not in names, "和 Rank 1 保护区重复计分的项应该已经砍掉"
    assert sum(full for _n, full in mg.SCORE_ITEMS) == 100, "主观项满分应该正好 100"

    # 主观项留空
    first = names.index(mg.SCORE_ITEMS[0][0])
    assert rows[first][2] == "" and rows[first][3] == ""
    # 求和公式的行号要正好圈住那几项（CSV 第 1 行是表头，行号从 1 数）
    total = next(row for row in rows if row[0].startswith("人工总分"))
    span = len(mg.SCORE_ITEMS)
    assert total[2] == "=SUM(C%d:C%d)" % (first + 1, first + span), total[2]
    assert "100" in total[1]


TESTS = (
    test_clean_montage_reports_nothing,
    test_low_score_material_is_rejected,
    test_duration_must_be_copied,
    test_word_time_must_be_copied,
    test_word_text_must_be_copied_verbatim,
    test_missing_word_is_reported,
    test_self_invented_word_is_reported,
    test_unknown_overlay_kind_is_rejected,
    test_rank1_climax_guard_is_hard,
    test_same_source_id_in_one_montage_is_rejected,
    test_reuse_across_montages_is_allowed,
    test_tail_insert_start_must_be_computed_not_guessed,
    test_gap_too_small_is_reported,
    test_every_pick_but_rank1_needs_a_comment,
    test_opening_must_match_and_sit_at_zero,
    test_tts_table_is_not_linear,
    test_side_by_side_objects_are_recognised,
    test_empty_reply_is_not_an_error,

    test_scorecard_fills_machine_part_and_leaves_manual_blank,
)




def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("")
    print("%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
