import streamlit as st
import pandas as pd
import calendar
from ortools.sat.python import cp_model

# --- 初期設定 ---
st.set_page_config(page_title="AIシフトメーカー", layout="wide")
st.title("📅 インタラクティブ・シフトメーカー")

# --- 設定サイドバー ---
with st.sidebar:
    st.header("⚙️ 基本設定")
    year = 2026
    month = st.selectbox("作成月", range(1, 13), index=3)
    max_h = st.number_input("上限時間(01-08)", value=177)
    s01_night_limit = st.number_input("スタ01夜勤上限", value=4)
    
    # 前月末からの引き継ぎ
    st.subheader("🌙 夜勤明け引き継ぎ (1日)")
    emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
    prev_n_end = st.multiselect("1日に夜勤明けになる2名を選択", emp_names, default=emp_names[:2])

# --- 1. 入力フォーム（データエディタ）の作成 ---
st.subheader("📝 勤務希望・確定事項の入力")
st.info("確定している勤務を選択してください。空欄の部分はAIが自動で埋めます。")

_, num_days = calendar.monthrange(year, month)
days = [f"{d}({['月','火','水','木','金','土','日'][calendar.weekday(year, month, d)]})" for d in range(1, num_days + 1)]

# 初期データの作成（すべて空欄）
init_df = pd.DataFrame(
    "",
    index=days,
    columns=emp_names
)

# データエディタを表示
edited_df = st.data_editor(
    init_df,
    use_container_width=True,
    column_config={
        col: st.column_config.SelectboxColumn(
            options=["", "日勤", "夜勤入", "休み"],
            width="small"
        ) for col in emp_names
    }
)

# --- 2. シフト計算ロジック ---
def solve_shift():
    model = cp_model.CpModel()
    all_days = range(1, num_days + 1)
    all_emps = range(len(emp_names))
    OFF, DAY, N_START, N_END = 0, 1, 2, 3
    STATES = [OFF, DAY, N_START, N_END]

    shifts = {}
    for e in all_emps:
        for d in all_days:
            for s in STATES:
                shifts[(e, d, s)] = model.NewBoolVar(f'e{e}d{d}s{s}')

    # --- 基本制約 ---
    for d in all_days:
        for e in all_emps:
            model.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) == 4)
        model.Add(sum(shifts[(e, d, N_START)] for e in all_emps) == 2)
        model.Add(sum(shifts[(e, d, N_END)] for e in all_emps) == 2)

    # UIでの入力内容をハード制約として追加
    for e_idx, e_name in enumerate(emp_names):
        for d_idx, d_label in enumerate(days):
            val = edited_df.iloc[d_idx, e_idx]
            day_num = d_idx + 1
            if val == "日勤": model.Add(shifts[(e_idx, day_num, DAY)] == 1)
            if val == "夜勤入": model.Add(shifts[(e_idx, day_num, N_START)] == 1)
            if val == "休み": model.Add(shifts[(e_idx, day_num, OFF)] == 1)

    # 1日の夜勤明け引き継ぎ
    for e_idx, e_name in enumerate(emp_names):
        if e_name in prev_n_end:
            model.Add(shifts[(e_idx, 1, N_END)] == 1)
        else:
            model.Add(shifts[(e_idx, 1, N_END)] == 0)

    # 共通ルール（夜勤明けの翌日、連勤制限、スタッフ別制限など）
    for e in all_emps:
        for d in range(1, num_days):
            model.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
            model.Add(shifts[(e, d, N_END)] + shifts[(e, d+1, DAY)] <= 1)
        for d in range(1, num_days - 5):
            model_ortools.Add(sum(shifts[(e, d + i, OFF)] for i in range(7)) >= 1)

    # --- 目的関数（等間隔・バランス） ---
    obj_terms = []
    for e in range(11):
        for d in all_days:
            obj_terms.append(shifts[(e, d, DAY)] * 10)
            if e < 7: obj_terms.append(shifts[(e, d, N_START)] * 15)
            
    # 高速な等間隔スコアリング（4日おき）
    for e in all_emps:
        for d in range(1, num_days - 4):
            is_sep = model.NewBoolVar(f'sep_e{e}d{d}')
            model.Add(is_sep <= shifts[(e, d, OFF)])
            model.Add(is_sep <= shifts[(e, d+4, OFF)])
            model.Add(is_sep >= shifts[(e, d, OFF)] + shifts[(e, d+4, OFF)] - 1)
            obj_terms.append(is_sep * 30)

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res = []
        for d in all_days:
            row = {"日付": days[d-1]}
            for e in all_emps:
                for s, name in {DAY:"日勤", N_START:"夜勤入", N_END:"夜勤明", OFF:"休み"}.items():
                    if solver.Value(shifts[(e, d, s)]): row[emp_names[e]] = name
            res.append(row)
        return pd.DataFrame(res).set_index("日付")
    return None

if st.button("🚀 AIに空欄を埋めてもらう"):
    with st.spinner("最適な組み合わせを計算中..."):
        result_df = solve_shift()
        if result_df is not None:
            st.success("シフトが完成しました！")
            st.dataframe(result_df, use_container_width=True)
            # 集計
            st.subheader("📊 勤務集計")
            st.table(result_df.apply(pd.Series.value_counts).fillna(0).astype(int))
        else:
            st.error("解が見つかりませんでした。入力した確定事項に無理がないか確認してください。")
