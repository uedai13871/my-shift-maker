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
    month = st.selectbox("作成月", range(1, 13), index=3)  # デフォルト4月
    max_h = st.number_input("上限時間(全スタッフ共通)", value=177)
    s01_night_limit = st.number_input("スタ01夜勤上限", value=4)
    st.info("※入力内容はアプリ内での操作中、保持されます。")

# --- 1. 入力データの保持ロジック (Session State) ---
emp_names = [f"スタッフ{str(i+1).zfill(2)}" for i in range(12)]
_, num_days = calendar.monthrange(year, month)
day_cols = [f"{d}" for d in range(1, num_days + 1)]

# セッション状態で入力を管理
state_key = f"df_input_{year}_{month}"
if state_key not in st.session_state:
    # 初めてその月を表示する場合、空の表を作成
    st.session_state[state_key] = pd.DataFrame("", index=emp_names, columns=day_cols)

# --- 2. 入力フォーム（データエディタ） ---
st.subheader("📝 勤務希望・確定事項の入力")
st.markdown("空欄はAIが自動で埋めます。入力内容は自動で一時保存されます。")

# データエディタを表示。編集されるたびに session_state を更新
edited_df = st.data_editor(
    st.session_state[state_key],
    use_container_width=True,
    key=f"editor_{state_key}", # keyを指定することで状態を紐付け
    column_config={
        col: st.column_config.SelectboxColumn(
            options=["", "日勤", "夜勤入", "夜勤明け", "休み"],
            width="small"
        ) for col in day_cols
    }
)

# 編集された内容をセッション状態に書き戻す
st.session_state[state_key] = edited_df

# --- 3. シフト計算ロジック ---
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

    # 基本制約（人数固定）
    for d in all_days:
        for e in all_emps:
            model.Add(sum(shifts[(e, d, s)] for s in STATES) == 1)
        model.Add(sum(shifts[(e, d, DAY)] for e in all_emps) >= 3)
        model.Add(sum(shifts[(e, d, N_START)] for e in all_emps) == 2)
        model.Add(sum(shifts[(e, d, N_END)] for e in all_emps) == 2)

    # ユーザー入力内容を反映
    for e_idx, e_name in enumerate(emp_names):
        for d_idx in range(num_days):
            val = edited_df.iloc[e_idx, d_idx]
            day_num = d_idx + 1
            if val == "日勤": model.Add(shifts[(e_idx, day_num, DAY)] == 1)
            if val == "夜勤入": model.Add(shifts[(e_idx, day_num, N_START)] == 1)
            if val == "夜勤明け": model.Add(shifts[(e_idx, day_num, N_END)] == 1)
            if val == "休み": model.Add(shifts[(e_idx, day_num, OFF)] == 1)

    # 共通ルール
    for e in all_emps:
        for d in range(1, num_days):
            model.Add(shifts[(e, d, N_START)] == shifts[(e, d+1, N_END)])
            model.Add(shifts[(e, d, N_END)] + shifts[(e, d+1, DAY)] <= 1)
        for d in range(1, num_days - 5):
            model.Add(sum(shifts[(e, d + i, OFF)] for i in range(7)) >= 1)
        hrs = [shifts[(e, d, DAY)]*8 + shifts[(e, d, N_START)]*6 + shifts[(e, d, N_END)]*8 for d in all_days]
        model.Add(sum(hrs) <= max_h)
    
    model.Add(sum(shifts[(0, d, N_START)] for d in all_days) <= s01_night_limit)
    
    for e in range(7, 12):
        for d in all_days:
            model.Add(shifts[(e, d, N_START)] == 0)
            model.Add(shifts[(e, d, N_END)] == 0)

    # 目的関数
    obj_terms = []
    # 15日目標
    for e in all_emps:
        work_vars = [shifts[(e, d, s)] for d in all_days for s in [DAY, N_START, N_END]]
        actual_work_days = sum(work_vars)
        under_15 = model.NewIntVar(0, 15, f'under15_e{e}')
        model.AddMinEquality(under_15, [actual_work_days, 15])
        obj_terms.append(under_15 * 100)
        obj_terms.append(actual_work_days * 10)

    # 休み等間隔
    for e in all_emps:
        for d in range(1, num_days - 4):
            is_sep = model.NewBoolVar(f'sep_e{e}d{d}')
            model.Add(is_sep <= shifts[(e, d, OFF)])
            model.Add(is_sep <= shifts[(e, d+4, OFF)])
            model.Add(is_sep >= shifts[(e, d, OFF)] + shifts[(e, d+4, OFF)] - 1)
            obj_terms.append(is_sep * 30)

    model.Maximize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers = 2
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res_data = []
        for e in all_emps:
            row = {"スタッフ": emp_names[e]}
            for d in all_days:
                for s, name in {DAY:"日勤", N_START:"夜勤入", N_END:"夜勤明", OFF:"休み"}.items():
                    if solver.Value(shifts[(e, d, s)]): row[f"{d}"] = name
            res_data.append(row)
        return pd.DataFrame(res_data).set_index("スタッフ")
    return None

# --- 4. 出力表示 ---
if st.button("🚀 シフトを作成する"):
    with st.spinner("AIが最適な組み合わせを計算中..."):
        result_df = solve_shift()
        if result_df is not None:
            st.success("シフトが完成しました！")
            # 背景色なしの標準表示
            st.dataframe(result_df, use_container_width=True)
            
            st.subheader("📊 最終集計")
            counts = result_df.T.apply(pd.Series.value_counts).fillna(0).astype(int)
            st.table(counts)
        else:
            st.error("解が見つかりませんでした。入力内容を確認してください。")
